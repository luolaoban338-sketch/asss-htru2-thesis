"""Independent boundary, result, decision, and model audit of complete resamples."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from src.data import load_dataset, digest, seed_for
from src.samplers import select_rows
from scripts.run_stability_pilot import restrict_to_pool
from scripts.run_joint_pilot import read, cached, save
from scripts.audit_joint_pilot import audit_selector, audit_model, independent_choice, assert_close, quota_check


def audit(run, replicate=None):
    cfg=read(run/'config.json'); joint=read(run/'joint_config.json'); base=read(run/'baseline_config.json')
    fp=read(run/'manifest.json')['fingerprint']
    for name,key in [('config.json','config'),('joint_config.json','joint_config'),
                     ('baseline_config.json','baseline'),('protocol.md','protocol'),('outer_splits.json','splits')]:
        assert digest(run/name)==fp[key]
    for rel,sha in fp['code'].items():
        assert digest(run/'code'/rel)==sha
        assert digest(ROOT/rel)==sha, 'Audit requires matching live implementation'
    original=load_dataset(joint['dataset']); assert original.sha256==fp['data']
    outer=read(run/'outer_splits.json')['folds'][cfg['outer_fold']]
    source=np.array(outer['train_positions']); test=np.array(outer['test_positions'])
    pool=cached(run/'development_pool.json'); assert pool is not None
    assert pool['original_positions']==source.tolist()
    assert pool['row_ids']==original.row_ids[source].tolist()
    assert pool['group_ids']==original.group_ids[source].tolist()
    assert pool['excluded_row_ids']==original.row_ids[test].tolist()
    assert pool['excluded_group_ids']==original.group_ids[test].tolist()
    assert not set(pool['row_ids']) & set(pool['excluded_row_ids'])
    assert not set(pool['group_ids']) & set(pool['excluded_group_ids'])
    data=restrict_to_pool(original,source); config={**joint,'seed_root':cfg['seed_root']}; d=len(data.X.columns)
    counts=dict(replicates=0,selectors=0,games=0,contribution_fits=0,constant_evaluations=0,
                inner_scoring_fits=0,final_classifier_fits=0,direct_utility_refits=0,arm_results=0)
    reps=range(cfg['replicates']) if replicate is None else [replicate]
    for r in reps:
        folder=run/f'replicate-{r}'; result=cached(folder/'result.json'); assert result is not None
        assert result['replicate']==r and result['outer_test_evaluated'] is False
        actual_files={p.relative_to(folder).as_posix() for p in folder.rglob('*.json')
                      if p.name not in ('result.json','result.complete.json')}
        assert actual_files==set(result['artifact_sha256'])
        for rel,sha in result['artifact_sha256'].items(): assert digest(folder/rel)==sha
        resample=cached(folder/'resample.json'); train=np.array(resample['positions'])
        assert len(train)==len(set(train)) and set(train)<=set(range(len(data.y)))
        assert resample['row_ids']==data.row_ids[train].tolist()
        assert resample['group_ids']==data.group_ids[train].tolist()
        seed=seed_for(cfg['seed_root'],data.name,r,'complete_resample'); assert resample['seed']==seed
        expected,_,_=select_rows(data.X,data.partition(np.arange(len(data.y)),'development_pool'),
             data.numeric,data.categorical,cfg['resample_rho'],seed,'stratified_random',base['sampler'])
        np.testing.assert_array_equal(train,expected)
        quota_check(data.y,data.y[train],cfg['resample_rho'])
        assert resample['pool_n']==len(data.y); assert_close(resample['actual_rho'],len(train)/len(data.y))
        inner=cached(folder/'inner_splits.json'); inner_seed=seed_for(cfg['seed_root'],data.name,r,'inner_split')
        assert inner['seed']==inner_seed and len(inner['folds'])==config['inner_folds']
        cv=StratifiedGroupKFold(n_splits=config['inner_folds'],shuffle=True,random_state=inner_seed)
        expected_splits=list(cv.split(data.X.iloc[train],data.y[train],data.group_ids[train]))
        validation=[]; all_rows=[]; replicate_counts={k:0 for k in result['counts']}
        for i,part in enumerate(inner['folds']):
            tr=np.array(part['training_positions']); va=np.array(part['validation_positions'])
            np.testing.assert_array_equal(tr,train[expected_splits[i][0]])
            np.testing.assert_array_equal(va,train[expected_splits[i][1]])
            assert set(tr)|set(va)==set(train) and not set(tr)&set(va)
            assert not set(data.group_ids[tr]) & set(data.group_ids[va])
            validation.extend(va.tolist()); dest=folder/f'inner-{i}'
            for method in config['methods']:
                counts['direct_utility_refits']+=audit_selector(dest/method,data,tr,va,config,base,r==0 and i==0)
                selector=cached(dest/method/'selector.json')
                replicate_counts['selectors']+=1; replicate_counts['games']+=config['B']
                replicate_counts['contribution_fits']+=selector['logistic_fits']
                replicate_counts['constant_evaluations']+=selector['constant_evaluations']
                for path in (dest/method/'models').glob('mask-*.json'):
                    if path.name.endswith('.complete.json'): continue
                    audit_model(path,data,tr,va,selector,base); replicate_counts['inner_scoring_fits']+=1
            rows=cached(dest/'candidates.json')
            expected_grid=[(m,lam,eta,k) for m in config['methods'] for lam in config['lambda_grid']
                           for eta in config['eta_grid'] for k in range(1,d+1)]
            assert len(rows)==len(expected_grid)
            for index,(row,expected_candidate) in enumerate(zip(rows,expected_grid)):
                assert row['candidate_id']==index
                assert tuple(row[k] for k in ['method','lam','eta','k'])==expected_candidate
                model=cached(dest/row['model_path']); selector=cached(dest/row['method']/'selector.json')
                assert digest(dest/row['model_path'])==row['model_sha256']
                path=next(p for p in selector['paths'] if p['lam']==row['lam'] and p['eta']==row['eta'])
                fs=[f for f in data.X.columns if f in path['path'][row['k']-1]['features']]
                assert fs==row['features']==model['context']['features']
                assert row['mask']==sum(1<<j for j,f in enumerate(data.X.columns) if f in fs)
                assert_close(row['f1'],model['metrics']['f1']); assert_close(row['actual_rho'],selector['actual_rho'])
            all_rows.append(rows)
        assert sorted(validation)==sorted(train)
        decisions=cached(folder/'decisions.json'); summaries=decisions['candidate_scores']
        assert len(summaries)==len(expected_grid)
        for index,row in enumerate(summaries):
            assert row['candidate_id']==index
            assert tuple(row[k] for k in ['method','lam','eta','k'])==expected_grid[index]
            matches=[rows[index] for rows in all_rows]
            assert_close(row['fold_f1'],[v['f1'] for v in matches])
            assert_close(row['mean_f1'],np.mean([v['f1'] for v in matches]))
            assert_close(row['mean_actual_rho'],np.mean([v['actual_rho'] for v in matches]))
            assert_close(row['cell_fraction'],row['mean_actual_rho']*row['k']/d)
        full=[v for v in summaries if v['method']=='full' and v['k']==d]
        assert_close([v['mean_f1'] for v in full],decisions['baseline_inner_f1'])
        threshold=full[0]['mean_f1']-config['epsilon']; tol=config['numeric_tolerance']
        assert_close(decisions['threshold'],threshold)
        assert decisions['feasible_candidate_ids']==[v['candidate_id'] for v in summaries if v['mean_f1']>=threshold-tol]
        choices=decisions['selections']
        restrictions={'full':full,'sample_only':[v for v in summaries if v['k']==d],
                      'feature_only':[v for v in summaries if v['method']=='full'],'joint':summaries}
        sample_id=independent_choice(restrictions['sample_only'],threshold,tol)
        restrictions['sequential']=[v for v in summaries if v['method']==summaries[sample_id]['method']]
        for label,method in [('random','random_half'),('ordinary','ordinary_half'),('robust','robust_half')]:
            restrictions['joint_'+label]=[v for v in summaries if v['method'] in ('full',method)]
        for arm,allowed in restrictions.items():
            winner=independent_choice(allowed,threshold,tol)
            assert choices[arm]==summaries[winner]
        for method in config['methods']:
            dest=folder/'refit'/method
            audit_selector(dest,data,train,np.array([],dtype=int),config,base,False)
            selector=cached(dest/'selector.json')
            replicate_counts['selectors']+=1; replicate_counts['games']+=config['B']
            replicate_counts['contribution_fits']+=selector['logistic_fits']
            replicate_counts['constant_evaluations']+=selector['constant_evaluations']
            assert not (dest/'models').exists()
        assert set(result['selections'])==set(config['comparisons'])
        for arm,out in result['selections'].items():
            c=choices[arm]; assert c==out['candidate']
            selector=cached(folder/'refit'/c['method']/'selector.json')
            path=next(p for p in selector['paths'] if p['lam']==c['lam'] and p['eta']==c['eta'])
            expected=[f for f in data.X.columns if f in path['path'][c['k']-1]['features']]
            assert out['features']==expected and len(expected)==c['k']
            assert out['selected_ids']==selector['selected_ids']
            assert out['selected_n']==len(out['selected_ids']) and out['resample_n']==len(train)
            assert not set(out['selected_ids']) & set(pool['excluded_row_ids'])
            assert_close(out['actual_rho'],out['selected_n']/len(train))
            assert_close(out['cell_fraction'],out['actual_rho']*len(expected)/d)
            counts['arm_results']+=1
        assert result['counts']==replicate_counts
        for key,value in replicate_counts.items(): counts[key]+=value
        counts['replicates']+=1
        print(f'AUDIT passed replicate={r}',flush=True)
    report={'status':'passed','run_id':run.name,**counts,'outer_test_evaluated':False,
            'scope':'all configured replicates' if replicate is None else f'replicate {replicate}',
            'manifest_sha256':digest(run/'manifest.json')}
    suffix='' if replicate is None else f'_rep{replicate}'
    save(ROOT/'qa'/f'{run.name}_audit{suffix}.json',report)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); ap.add_argument('--replicate',type=int)
    args=ap.parse_args()
    with threadpool_limits(limits=1): print(audit(ROOT/'runs'/args.run_id,args.replicate))
