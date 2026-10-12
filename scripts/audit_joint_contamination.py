"""Independent perturbation reconstruction plus full selector/model audits."""
from pathlib import Path
import argparse
from dataclasses import replace
from math import ceil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.data import load_dataset,digest,seed_for
from scripts.run_joint_pilot import read,save,cached
from scripts.run_joint_contamination import setup,frame_hash,verify_clean
from scripts.run_contamination import conditions
from scripts.audit_joint_pilot import audit_selector,audit_model,assert_close

def reconstruct(data,train,rate,seed,multiplier):
    """No call to production contaminate or perturbed_data."""
    original=data.X.iloc[train]; X=data.X.copy(deep=True)
    scales={}
    for f in data.numeric:
        values=original[f].dropna().to_numpy(float)
        lo,hi=np.percentile(values,[25,75],method='linear')
        if hi>lo: scales[f]=float(hi-lo)
    ids=data.row_ids[train]; count=ceil(rate*len(train))
    positions=np.random.default_rng(seed).permutation(np.argsort(ids,kind='stable'))[:count]
    events=[]
    for local in positions:
        eligible=[f for f in scales if np.isfinite(float(original.iloc[local][f]))]
        rng=np.random.default_rng(seed_for(seed,'cell',ids[local]))
        f=eligible[int(rng.integers(len(eligible)))]; sign=int(rng.choice([-1,1]))
        before=float(original.iloc[local][f]); after=before+sign*multiplier*scales[f]
        X.iloc[train[local],X.columns.get_loc(f)]=after
        events.append({'local_position':int(local),'row_id':str(ids[local]),'feature':f,'before':before,'after':after,'sign':sign,'iqr':scales[f]})
    return replace(data,X=X),events,scales

def audit_condition(run,config,joint,base,data,ref,fold,rate,draw,label):
    folder=run/f'fold-{fold}'/label
    complete=cached(folder/'condition_complete.json'); assert complete is not None
    for rel,h in complete['sha256'].items(): assert digest(folder/rel)==h,rel
    part=read(run/'outer_splits.json')['folds'][fold]
    train=np.array(part['train_positions']); test=np.array(part['test_positions'])
    seed=seed_for(config['seed_root'],data.name,fold,draw,'contamination')
    changed,events,scales=reconstruct(data,train,rate,seed,config['iqr_multiplier'])
    pollution=cached(folder/'contamination.json')
    assert pollution['events']==events and pollution['count']==ceil(rate*len(train))
    assert pollution['eligible_iqr']==scales and pollution['seed']==seed
    assert pollution['rate']==rate and pollution['multiplier']==config['iqr_multiplier']
    assert_close(pollution['actual_rate'],len(events)/len(train))
    assert int((changed.X.to_numpy()!=data.X.to_numpy()).sum())==len(events)
    np.testing.assert_array_equal(changed.X.iloc[test],data.X.iloc[test])
    np.testing.assert_array_equal(changed.y,data.y)
    context=cached(folder/'context.json')
    assert context['clean_test_hash']==frame_hash(data.X.iloc[test])
    assert context['perturbed_train_hash']==frame_hash(changed.X.iloc[train])
    assert context['train_ids']==data.row_ids[train].tolist() and context['test_ids']==data.row_ids[test].tolist()
    assert context['contamination_sha256']==digest(folder/'contamination.json')
    assert context['decisions_sha256']==digest(run/'frozen_decisions'/f'fold-{fold}.json')
    if rate==.1:
        small=cached(run/f'fold-{fold}'/f'rate-05-draw-{draw}'/'contamination.json')
        assert events[:len(small['events'])]==small['events']
    choices=read(run/'frozen_decisions'/f'fold-{fold}.json')['selections']
    methods={c['method'] for c in choices.values()}; selectors={}; utility_refits=0; fits=0; constants=0; utility_validation_changed=0
    for method in methods:
        dest=folder/'refit'/method
        utility_refits+=audit_selector(dest,changed,train,test,joint,base,True)
        s=cached(dest/'selector.json'); selectors[method]=s
        fits+=s['logistic_fits']; constants+=s['constant_evaluations']
        # Independent split/subsample seed and method consistency checks.
        assert s['method']==next(m for m in base['methods'] if m['name']==method)
        assert s['partition_id']==f'{data.name}:outer{fold}:refit'
        assert s['sample_seed']==seed_for(joint['seed_root'],s['partition_id'],'selector_sampler')
        altered_ids={e['row_id'] for e in events}
        for b in range(joint['B']):
            g=cached(dest/f'game-{b}.json')
            assert g['subsample_seed']==seed_for(joint['seed_root'],s['partition_id'],b,'feature_resample')
            assert g['game_seed']==seed_for(joint['seed_root'],s['partition_id'],b,'utility')
            assert g['permutation_seed']==seed_for(joint['seed_root'],s['partition_id'],method,b,'permutations')
            utility_validation_changed+=sum(len(altered_ids & set(f['validation_ids'])) for f in g['context']['folds'])
    if rate: assert utility_validation_changed>0
    model_paths=set(); rows=[]
    for arm,candidate in choices.items():
        r=cached(folder/'outer'/f'{arm}.json'); assert r is not None
        assert r['candidate']==candidate
        assert r['decisions_sha256']==context['decisions_sha256']
        s=selectors[candidate['method']]
        path=next(p for p in s['paths'] if p['lam']==candidate['lam'] and p['eta']==candidate['eta'])
        fs=[f for f in data.X.columns if f in path['path'][candidate['k']-1]['features']]
        assert r['features']==fs and len(fs)==candidate['k']
        assert r['selected_n']==len(s['selected_ids']) and r['train_n']==len(train) and r['test_n']==len(test)
        assert_close(r['actual_rho'],r['selected_n']/len(train)); assert_close(r['cell_fraction'],r['actual_rho']*len(fs)/8)
        mp=folder/r['model_path']; assert digest(mp)==r['model_sha256']
        m=audit_model(mp,changed,train,test,s,base)
        assert m['metrics']==r['metrics'] and m['context']['features']==fs
        assert m['context']['model_seed']==seed_for(base['seed_root'],data.name,fold,'classifier')
        model_paths.add(mp)
        clean=cached(run/f'fold-{fold}'/'rate-00-draw-0'/'outer'/f'{arm}.json')
        clean_model=cached(run/f'fold-{fold}'/'rate-00-draw-0'/clean['model_path'])
        a=set(clean['features']); bset=set(fs); ar=set(clean_model['context']['selected_ids']); br=set(s['selected_ids'])
        rows.append({'fold':fold,'rate':rate,'draw':draw,'arm':arm,'method':candidate['method'],'k':len(fs),
            'f1':r['metrics']['f1'],'roc_auc':r['metrics']['roc_auc'],'f1_loss':clean['metrics']['f1']-r['metrics']['f1'],
            'auc_loss':clean['metrics']['roc_auc']-r['metrics']['roc_auc'],'feature_jaccard':len(a&bset)/len(a|bset),
            'row_jaccard':len(ar&br)/len(ar|br),'actual_rho':r['actual_rho'],'cell_fraction':r['cell_fraction']})
    if rate==0: verify_clean(folder,ref,fold,choices)
    return {'selectors':len(selectors),'games':len(selectors)*joint['B'],'contribution_fits':fits,
        'final_model_fits':len(model_paths),'constant_evaluations':constants,'direct_utility_refits':utility_refits,
        'changed_utility_validation_occurrences':utility_validation_changed,'arm_outputs':8},rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); ap.add_argument('--fold',type=int,required=True,choices=range(5)); ap.add_argument('--clean-only',action='store_true')
    args=ap.parse_args(); run,config,joint,base,data,ref=setup(args.run_id)
    counts={}; rows=[]; n=0
    with threadpool_limits(limits=1):
        for rate,draw,label in conditions(config):
            if args.clean_only and rate: continue
            c,r=audit_condition(run,config,joint,base,data,ref,args.fold,rate,draw,label)
            for key,v in c.items(): counts[key]=counts.get(key,0)+v
            rows+=r; n+=1
            print(f'AUDIT passed fold={args.fold} {label}',flush=True)
    suffix='_clean' if args.clean_only else ''
    save(ROOT/'qa'/f'{args.run_id}_audit_fold{args.fold}{suffix}.json',{'status':'passed','fold':args.fold,'conditions':n,'counts':counts,'rows':rows,'run_manifest_sha256':digest(run/'manifest.json')})
    print(counts,flush=True)

if __name__=='__main__': main()
