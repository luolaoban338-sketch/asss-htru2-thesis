"""Independent CSR formulas, identity boundaries, utility refits and prediction audit."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from src.data import digest
from src.samplers import select_rows
from src.preprocessing import TabularEncoder
from experiments.feature_utility import FeatureUtility
from experiments.feature_path import redundancy_matrix
from experiments.ablation import seed,Budget
from scripts.audit_joint_pilot import audit_game,audit_model,quota_check,assert_close
from scripts.run_joint_pilot import cached
from scripts.run_ablation import setup,package_path,training_positions,ensure,check_stop,MAIN,REPLAY


def independent_paths(fs,phi,mc,R,cfg):
    mean=phi.mean(0); observed=phi.var(0,ddof=1); noise=mc.mean(0); raw=observed-noise
    outputs={}
    for label in cfg['logical_csr']:
        c,s,r=map(int,label)
        variance=np.clip(raw,0,None) if c else observed
        scores=mean-cfg['lambda_enabled']*np.sqrt(variance) if s else mean
        selected=[]; steps=[]
        for _ in fs:
            remaining=[j for j in range(len(fs)) if j not in selected]
            gains={j:float(scores[j]-(cfg['eta_enabled'] if r else 0)/len(fs)*sum(R[j,i] for i in selected)) for j in remaining}
            winner=sorted(remaining,key=lambda j:(-gains[j],j))[0]
            selected.append(winner)
            objective=sum(scores[j] for j in selected)-(cfg['eta_enabled'] if r else 0)/len(fs)*sum(R[i,j] for i in selected for j in selected if i<j)
            steps.append({'features':[fs[j] for j in selected],'increment':gains[winner],'objective':objective})
        outputs[label]={'scores':scores,'steps':steps,'features':[f for f in fs if f in steps[cfg['k']-1]['features']]}
    return {'mean':mean,'observed_variance':observed,'mean_mc_variance':noise,'raw_corrected_variance':raw,
        'corrected_sd':np.sqrt(np.clip(raw,0,None)),'clipped_to_zero':raw<0,'score':mean-cfg['lambda_enabled']*np.sqrt(np.clip(raw,0,None))},outputs


def audit(run,cfg,base,data,splits,phase,root,unit,budget):
    folder=package_path(run,phase,root,unit)
    old=cached(folder/'audit.json')
    if old is not None:
        for rel,h in old['sha256'].items(): assert digest(folder/rel)==h
        print(f'AUDIT SKIP {phase} {unit} {root}',flush=True); return old
    package=cached(folder/'package.json'); assert package is not None
    train,excluded=training_positions(data,splits,cfg,phase,root,unit)
    assert cached(folder/'context.json')=={'phase':phase,'root':root,'unit':unit,'train_positions':train.tolist(),'excluded_positions':excluded.tolist()}
    fs=list(data.X.columns); selectors={}; games_n=0; audited_models=set()
    rng=np.random.default_rng(seed(cfg,phase,root,unit,'shared','random_features'))
    expected_random=set(rng.choice(8,4,replace=False).tolist())
    for name in cfg['samplers']:
        check_stop(); dest=folder/name; s=cached(dest/'selector.json'); assert s is not None
        selectors[name]=s; method=next(m for m in base['methods'] if m['name']==name)
        assert s['method']==method and s['training_positions']==train.tolist()
        assert s['training_ids']==data.row_ids[train].tolist()
        chosen=np.array(s['selected_positions']); assert set(chosen)<=set(train) and len(chosen)==len(set(chosen))
        assert s['selected_ids']==data.row_ids[chosen].tolist()
        quota_check(data.y[train],data.y[chosen],cfg['rho'])
        assert s['sample_seed']==seed(cfg,phase,root,unit,name,'final_rows')
        local,_,diagnostic=select_rows(data.X.iloc[train],data.partition(train,s['partition_id']),data.numeric,data.categorical,cfg['rho'],s['sample_seed'],method['strategy'],base['sampler'])
        assert chosen.tolist()==train[local].tolist()
        # Serialized class-quota keys are strings; compare through serial conversion.
        from scripts.run_feature_pilot import serial
        assert s['sampler_diagnostics']==json.loads(json.dumps(serial(diagnostic)))
        R=redundancy_matrix(data.X.iloc[chosen],data.numeric,data.categorical); assert_close(R,s['redundancy'])
        games=[]
        for b in range(cfg['B']):
            check_stop(); g=cached(dest/f'game-{b}.json'); assert g is not None
            assert digest(dest/f'game-{b}.json')==s['game_sha256'][f'game-{b}.json']
            assert g['subsample_seed']==seed(cfg,phase,root,unit,'shared','contribution_subsample',b)
            assert g['game_seed']==seed(cfg,phase,root,unit,'shared','game',b)
            assert g['permutation_seed']==seed(cfg,phase,root,unit,name,'permutation',b)
            local,_,_=select_rows(data.X.iloc[train],data.partition(train,s['partition_id']),data.numeric,data.categorical,cfg['contribution_subsample_rho'],g['subsample_seed'],'stratified_random',base['sampler'])
            pool=train[local]; assert pool.tolist()==g['subsample_positions']
            rebuilt=FeatureUtility(data.X.iloc[pool],data.partition(pool,f'{s["partition_id"]}:resample{b}'),data.numeric,data.categorical,method,base,g['game_seed'],cfg['utility_folds'])
            assert rebuilt.context_hash==g['context_hash']
            audit_game(g,data,train,excluded,method,{**cfg,'subsample_rho':cfg['contribution_subsample_rho']})
            assert_close(np.sqrt(np.mean((np.array(g['permutation']['phi'])-g['exact_phi'])**2)),g['rmse'])
            assert_close(np.max(np.abs(np.array(g['permutation']['phi'])-g['exact_phi'])),g['max_abs_error'])
            # One predetermined nonempty subset-fold per main or replay game.
            refit_path=dest/f'audit-refit-{b}.json'
            if cached(refit_path) is None:
                mask=b+1; fold=b%3; f=rebuilt.folds[fold]; cols=[f for j,f in enumerate(fs) if mask & (1<<j)]
                enc=TabularEncoder(cols,()); Z=enc.fit_transform(rebuilt.X.iloc[f['selected']][cols])
                params={k:v for k,v in base['model'].items() if k!='class'}
                model=LogisticRegression(**params,random_state=f['model_seed']).fit(Z,rebuilt.y[f['selected']])
                prob=model.predict_proba(enc.transform(rebuilt.X.iloc[f['validation']][cols]))[:,list(model.classes_).index(1)]
                value=float(f1_score(rebuilt.y[f['validation']],prob>=base['threshold'],zero_division=0))
                expected=next(c for c in g['coalitions'] if c['mask']==mask)['fold_f1'][fold]
                assert_close(value,expected)
                ensure(refit_path,{'status':'passed','mask':mask,'utility_fold':fold,'f1':value,'game_sha256':digest(dest/f'game-{b}.json')})
            else:
                assert cached(refit_path)['game_sha256']==digest(dest/f'game-{b}.json')
            games.append(g); games_n+=1
        phi=np.array([g['permutation']['phi'] for g in games]); mc=np.array([g['permutation']['mc_variance_of_mean'] for g in games])
        stats,paths=independent_paths(fs,phi,mc,R,cfg)
        for key,value in stats.items(): assert_close(value,s['statistics'][key])
        _,exact_paths=independent_paths(fs,np.array([g['exact_phi'] for g in games]),mc,R,cfg)
        for label,expected in paths.items():
            actual=s['variants'][label]; assert actual['features']==expected['features']
            assert_close(actual['scores'],expected['scores']); assert len(actual['path'])==8
            for k,(a,e) in enumerate(zip(actual['path'],expected['steps']),1):
                assert a['k']==k and a['features']==e['features'] and a['added']==e['features'][-1]
                assert_close(a['increment'],e['increment']); assert_close(a['objective'],e['objective'])
            assert s['exact_prefix_agreement'][label]==(expected['steps'][3]['features']==exact_paths[label]['steps'][3]['features'])
        for alias,canonical in cfg['aliases'].items(): assert s['variants'][alias]==s['variants'][canonical]
        assert s['random_features']==[f for j,f in enumerate(fs) if j in expected_random]
        assert_close(s['actual_rho'],len(chosen)/len(train))
    records=package['records']; expected_keys={(n,l) for n in cfg['samplers'] for l in cfg['logical_csr']+['random_four']}
    if phase=='predictive': expected_keys|={(n,'half_all') for n in cfg['samplers']}|{('full','full_all')}
    assert len(records)==len(expected_keys) and {(r['sampler'],r['label']) for r in records}==expected_keys
    barrier=cached(folder/'selection_barrier.json')
    assert barrier=={'sha256':{n:digest(folder/n/'selector.json') for n in cfg['samplers']}}
    for row in records:
        name,label=row['sampler'],row['label']
        s=selectors[name] if name!='full' else {'selected_positions':train.tolist(),'selected_ids':data.row_ids[train].tolist(),'actual_rho':1.}
        expected=(fs if label in ('half_all','full_all') else s['random_features'] if label=='random_four' else s['variants'][label]['features'])
        assert row['features']==expected and row['selected_n']==len(s['selected_ids']) and row['train_n']==len(train)
        assert_close(row['actual_rho'],s['actual_rho']); assert_close(row['cell_fraction'],s['actual_rho']*len(expected)/8)
        if phase=='predictive':
            path=folder/row['model_path']; assert digest(path)==row['model_sha256']
            model=cached(path); assert model['context']['features']==expected
            assert model['context']['model_seed']==seed(cfg,phase,root,unit,'shared','final_model')
            assert model['metrics']==row['metrics']
            if row['model_path'] not in audited_models:
                audit_model(path,data,train,excluded,s,base); audited_models.add(row['model_path'])
        else: assert 'metrics' not in row and 'model_path' not in row
    if phase=='stability': assert not list(folder.rglob('mask-*.json'))
    files={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*.json') if p.name not in ('audit.json','audit.complete.json')}
    result=ensure(folder/'audit.json',{'status':'passed','phase':phase,'unit':unit,'root':root,'selectors':3,'games':games_n,
        'logical_records':len(records),'reconstructed_final_models':len(audited_models),'direct_utility_refits':games_n,'sha256':files})
    print(f'AUDIT PASS {phase} {unit} {root}: games={games_n}, models={len(audited_models)}, records={len(records)}',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',choices=[MAIN,REPLAY],default=MAIN)
    ap.add_argument('--phase',choices=['predictive','stability'],required=True); ap.add_argument('--unit',type=int,required=True); ap.add_argument('--root',type=int,default=2026100501)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run,cfg,base,data,splits=setup(args.run_id)
        budget=Budget(ROOT/'runs'/MAIN/'fit_budget.sqlite3')
        with budget.instrument(),budget.track(f'audit:{run.name}:{args.phase}:{args.unit}:{args.root}'):
            audit(run,cfg,base,data,splits,args.phase,args.root,args.unit,budget)
        print(budget.counts(),flush=True)

if __name__=='__main__': main()
