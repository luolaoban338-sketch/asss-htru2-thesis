"""Frozen HTRU2 fixed-budget experiment; one CLI invocation is one package."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['LOKY_MAX_CPU_COUNT']='1'
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from src.data import load_dataset,digest
from src.contracts import assert_disjoint
from src.samplers import select_rows
from experiments.feature_utility import FeatureUtility
from experiments.feature_path import redundancy_matrix
from experiments.shapley import permutation_shapley,exact_shapley
from experiments.ablation import seed,seed_ledger,variants,Budget
from scripts.run_joint_pilot import read,save,cached,commit,fit_evaluate

DESIGN=ROOT/'protocols/ablation-htru2-fixedbudget-v1-20261005'
MAIN='ablation-htru2-fixedbudget-20261005-v1'
REPLAY='ablation-htru2-fixedbudget-replay-20261005-v1'
QA=ROOT/'qa/ablation_20261006'


def check_stop():
    if (ROOT/'STOP_ABLATION').exists(): raise SystemExit(75)


def ensure(path,value):
    from scripts.run_feature_pilot import serial
    old=cached(path)
    if old is not None:
        if old!=serial(value): raise RuntimeError(f'Completed artifact changed: {path}')
        return old
    return commit(path,value)


def verify_tree(folder):
    count=0
    for marker in folder.rglob('*.complete.json'):
        path=marker.with_name(marker.name.replace('.complete.json','.json'))
        assert cached(path) is not None
        count+=1
    return count


def setup(run_id,initialize=False):
    if run_id not in (MAIN,REPLAY): raise ValueError('Use the prespecified main or replay run')
    config=read(DESIGN/'config.json'); frozen=read(DESIGN/'freeze_manifest.json')
    for section in ('design_sha256','inherited_sha256'):
        for rel,h in frozen[section].items():
            if digest(ROOT/rel)!=h: raise RuntimeError(f'Frozen inherited file changed: {rel}')
    base=read(ROOT/'configs/baseline_pilot_v1.json')
    assert {k:v for k,v in base['model'].items() if k!='class'}==config['model']
    data=load_dataset('htru2')
    assert len(data.y)==17898 and len(data.X.columns)==8 and not data.categorical
    assert len(set(data.group_ids))==len(data.y), 'Singleton fingerprint groups changed'
    splits=read(ROOT/f"runs/{config['source_baseline_run']}/htru2/splits.json")
    all_test=[]
    for f in splits['folds']:
        tr=np.array(f['train_positions']); te=np.array(f['test_positions'])
        assert set(tr)|set(te)==set(range(len(data.y)))
        assert_disjoint(data.partition(tr,'preflight'),data.row_ids[te],data.group_ids[te])
        all_test.extend(te.tolist())
    assert sorted(all_test)==list(range(len(data.y)))
    paths=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'experiments').glob('*.py'))
    paths += [ROOT/'scripts'/n for n in ('run_baselines.py','run_feature_pilot.py','run_joint_pilot.py','audit_joint_pilot.py','run_ablation.py','audit_ablation.py')]
    paths += [ROOT/'tests/test_ablation.py']
    fingerprint={'design_manifest':digest(DESIGN/'freeze_manifest.json'),
        'code':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'python':sys.version,'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl')},
        'numeric_libraries':[{k:v for k,v in info.items() if k not in ('num_threads','filepath')} for info in threadpool_info()]}
    run=ROOT/'runs'/run_id
    if not (run/'manifest.json').exists():
        if not initialize: raise RuntimeError('Initialize after implementation gates before fitting')
        gates=read(QA/'implementation_gates.json')
        assert gates['status']=='passed' and gates['code_sha256']==fingerprint['code']
        run.mkdir(parents=True,exist_ok=False)
        for p in paths:
            dest=run/'code'/p.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
        ensure(run/'config.json',config); ensure(run/'outer_splits.json',splits)
        ensure(run/'seed_ledger.json',seed_ledger(config))
        ensure(run/'manifest.json',{'fingerprint':fingerprint,'created_utc':datetime.now(timezone.utc).isoformat(),
            'implementation_gate_sha256':digest(QA/'implementation_gates.json'),
            'budget_database':str(ROOT/'runs'/MAIN/'fit_budget.sqlite3'), 'automatic_resume':False})
    assert cached(run/'manifest.json')['fingerprint']==fingerprint, 'Execution fingerprint changed; new run required'
    for rel,h in fingerprint['code'].items(): assert digest(run/'code'/rel)==h
    assert cached(run/'config.json')==config and cached(run/'outer_splits.json')==splits
    assert cached(run/'seed_ledger.json')==seed_ledger(config)
    verify_tree(run)
    return run,config,base,data,splits


def package_path(run,phase,root,unit):
    return run/phase/f'unit-{unit}-seed-{root}'


def training_positions(data,splits,config,phase,root,unit):
    outer=splits['folds'][unit if phase=='predictive' else 0]
    train=np.array(outer['train_positions']); excluded=np.array(outer['test_positions'])
    if phase=='stability':
        rseed=seed(config,phase,config['stability']['resample_root_seeds'][unit],unit,'shared','data_resample')
        local,_,_=select_rows(data.X.iloc[train],data.partition(train,'stability_pool'),data.numeric,data.categorical,
            config['stability']['resample_rho'],rseed,'stratified_random')
        train=train[local]
    return train,excluded


def selector(data,train,config,base,phase,root,unit,method,dest,budget):
    old=cached(dest/'selector.json')
    if old is not None: return old
    part_id=f'{phase}:{root}:{unit}'
    part=data.partition(train,part_id)
    rows_seed=seed(config,phase,root,unit,method['name'],'final_rows')
    with budget.track(f'main:{phase}:{unit}:{root}:{method["name"]}:rows'):
        local,_,diagnostics=select_rows(data.X.iloc[train],part,data.numeric,data.categorical,config['rho'],rows_seed,method['strategy'],base['sampler'])
        chosen=train[local]
        R=redundancy_matrix(data.X.iloc[chosen],data.numeric,data.categorical)
    games=[]
    for b in range(config['B']):
        check_stop(); path=dest/f'game-{b}.json'; game_result=cached(path)
        if game_result is None:
            t=time.perf_counter()
            sub_seed=seed(config,phase,root,unit,'shared','contribution_subsample',b)
            game_seed=seed(config,phase,root,unit,'shared','game',b)
            perm_seed=seed(config,phase,root,unit,method['name'],'permutation',b)
            with budget.track(f'{dest.parents[2].name}:{phase}:{unit}:{root}:{method["name"]}:game-{b}'):
                local,_,_=select_rows(data.X.iloc[train],part,data.numeric,data.categorical,config['contribution_subsample_rho'],sub_seed,'stratified_random',base['sampler'])
                pool=train[local]
                game=FeatureUtility(data.X.iloc[pool],data.partition(pool,f'{part_id}:resample{b}'),data.numeric,data.categorical,method,base,game_seed,config['utility_folds'])
                approx=permutation_shapley(game.utility,game.features,config['M'],perm_seed)
                permutation_fits=game.fit_count
                exact=exact_shapley(game.utility,game.features)
            game_result=ensure(path,{'resample':b,'subsample_positions':pool,'subsample_seed':sub_seed,'game_seed':game_seed,
                'permutation_seed':perm_seed,'context':game.context,'context_hash':game.context_hash,'features':game.features,
                'permutation':approx,'exact_phi':exact,'max_abs_error':float(np.abs(approx['phi']-exact).max()),
                'rmse':float(np.sqrt(np.mean((approx['phi']-exact)**2))), 'coalitions':game.cache_records(),
                'permutation_logistic_fits':permutation_fits,'logistic_fits':game.fit_count,'constant_evaluations':game.constant_count,
                'seconds':time.perf_counter()-t})
            print(f'GAME {phase} unit={unit} root={root} {method["name"]} b={b} fits={game.fit_count} seconds={time.perf_counter()-t:.1f}',flush=True)
        games.append(game_result)
    phi=np.array([g['permutation']['phi'] for g in games]); mc=np.array([g['permutation']['mc_variance_of_mean'] for g in games])
    stats,paths=variants(list(data.X.columns),phi,mc,R,config)
    # Exact diagnostic uses the same estimated MC correction only to isolate contribution-estimation error.
    _,exact_paths=variants(list(data.X.columns),np.array([g['exact_phi'] for g in games]),mc,R,config)
    rng=np.random.default_rng(seed(config,phase,root,unit,'shared','random_features'))
    random_set=set(rng.choice(8,4,replace=False).tolist())
    return ensure(dest/'selector.json',{'partition_id':part_id,'phase':phase,'root_seed':root,'unit':unit,'method':method,
        'training_positions':train,'training_ids':data.row_ids[train], 'selected_positions':chosen,'selected_ids':data.row_ids[chosen],
        'sample_seed':rows_seed,'sampler_diagnostics':diagnostics,'features':list(data.X.columns),'redundancy':R,
        'actual_rho':len(chosen)/len(train),'statistics':stats,'variants':paths,
        'random_features':[f for j,f in enumerate(data.X.columns) if j in random_set],
        'exact_prefix_agreement':{label:paths[label]['path'][3]['features']==exact_paths[label]['path'][3]['features'] for label in config['logical_csr']},
        'exact_prefix_definition':'ordered k=4 path; exact phi substituted with same estimated MC variance for diagnostic only',
        'game_sha256':{f'game-{b}.json':digest(dest/f'game-{b}.json') for b in range(config['B'])}})


def execute(run,config,base,data,splits,phase,root,unit,budget):
    if phase=='predictive':
        assert unit in config['outer_folds'] and root in config['predictive_root_seeds']
    else:
        assert unit in range(10) and root==config['stability']['algorithm_root_seed']
    if run.name==REPLAY:
        assert (phase=='predictive' and unit==0 and root==config['replay']['predictive_root_seed']) or (phase=='stability' and unit==0)
    first=package_path(ROOT/'runs'/MAIN,'predictive',config['predictive_root_seeds'][0],0)
    if not (run.name==MAIN and phase=='predictive' and unit==0 and root==config['predictive_root_seeds'][0]):
        gate=cached(first/'audit.json')
        assert gate and gate['status']=='passed', 'First predictive package audit is mandatory'
    folder=package_path(run,phase,root,unit)
    if cached(folder/'package.json') is not None:
        print(f'SKIP {phase} unit={unit} root={root}',flush=True); return
    train,excluded=training_positions(data,splits,config,phase,root,unit)
    assert_disjoint(data.partition(train,'selector'),data.row_ids[excluded],data.group_ids[excluded])
    ensure(folder/'context.json',{'phase':phase,'root':root,'unit':unit,'train_positions':train,'excluded_positions':excluded})
    selectors={}
    for name in config['samplers']:
        check_stop(); method=next(m for m in base['methods'] if m['name']==name)
        selectors[name]=selector(data,train,config,base,phase,root,unit,method,folder/name,budget)
    # Selection barrier: all three selector files are durable before any outer-model fit.
    ensure(folder/'selection_barrier.json',{'sha256':{name:digest(folder/name/'selector.json') for name in config['samplers']}})
    records=[]
    for name,s in selectors.items():
        arms={label:s['variants'][label]['features'] for label in config['logical_csr']}
        arms['random_four']=s['random_features']
        if phase=='predictive': arms['half_all']=list(data.X.columns)
        for label,features in arms.items():
            check_stop()
            record={'sampler':name,'label':label,'features':features,'actual_rho':s['actual_rho'],
                'selected_n':len(s['selected_ids']),'train_n':len(train),'cell_fraction':s['actual_rho']*len(features)/8}
            if phase=='predictive':
                mask=sum(1<<list(data.X.columns).index(f) for f in features)
                path=folder/name/'models'/f'mask-{mask}.json'
                with budget.track(f'{run.name}:{phase}:{unit}:{root}:final'):
                    model=fit_evaluate(data,train,s['selected_positions'],excluded,features,base,seed(config,phase,root,unit,'shared','final_model'),path)
                record.update(metrics=model['metrics'],model_path=path.relative_to(folder).as_posix(),model_sha256=digest(path))
            records.append(record)
    if phase=='predictive':
        check_stop(); path=folder/'full/models/mask-255.json'
        with budget.track(f'{run.name}:{phase}:{unit}:{root}:final'):
            model=fit_evaluate(data,train,train,excluded,tuple(data.X.columns),base,seed(config,phase,root,unit,'shared','final_model'),path)
        records.append({'sampler':'full','label':'full_all','features':list(data.X.columns),'actual_rho':1.,'cell_fraction':1.,
            'selected_n':len(train),'train_n':len(train),'metrics':model['metrics'],'model_path':path.relative_to(folder).as_posix(),'model_sha256':digest(path)})
    assert len(records)==(31 if phase=='predictive' else 27)
    ensure(folder/'package.json',{'status':'computed_pending_audit','phase':phase,'root':root,'unit':unit,'records':records})
    print(f'PACKAGE {phase} unit={unit} root={root}: {len(records)} logical records',flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',default=MAIN,choices=[MAIN,REPLAY]); ap.add_argument('--initialize',action='store_true')
    ap.add_argument('--phase',choices=['predictive','stability']); ap.add_argument('--unit',type=int); ap.add_argument('--root',type=int,default=2026100501)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run,config,base,data,splits=setup(args.run_id,args.initialize)
        if args.phase:
            check_stop()
            budget=Budget(ROOT/'runs'/MAIN/'fit_budget.sqlite3')
            with budget.instrument(): execute(run,config,base,data,splits,args.phase,args.root,args.unit,budget)
            print(json.dumps(budget.counts()),flush=True)

if __name__=='__main__': main()
