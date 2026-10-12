"""Frozen-budget contamination, durable per-game cache, and strict clean equivalence."""
from pathlib import Path
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import digest, load_dataset, seed_for
from experiments.contamination import contaminate
from experiments.joint_selection import subset_for
from scripts.run_joint_pilot import read, save, cached, commit, build_selector, fit_evaluate
from scripts.check_joint_replay import stable
from scripts.run_contamination import conditions

def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False).values.tobytes()).hexdigest()

def perturbed_data(data, train, test, rate, seed, multiplier):
    if set(train) & set(test) or set(train) | set(test) != set(range(len(data.X))):
        raise ValueError('Outer partition mismatch')
    X, events = contaminate(data.X.iloc[train], data.row_ids[train], data.numeric, rate, seed, multiplier)
    all_X = data.X.copy(deep=True); all_X.iloc[train] = X.to_numpy()
    changed = replace(data, X=all_X)
    if frame_hash(changed.X.iloc[test]) != frame_hash(data.X.iloc[test]):
        raise AssertionError('Test data changed')
    return changed, events

def frozen_choices(decisions, joint):
    choices = decisions['selections']
    if set(choices) != set(joint['comparisons']): raise ValueError('Arm mismatch')
    for c in choices.values():
        if c['method'] not in joint['methods'] or c['lam'] not in joint['lambda_grid'] or c['eta'] not in joint['eta_grid'] or not 1 <= c['k'] <= 8:
            raise ValueError('Invalid frozen decision')
    return json.loads(json.dumps(choices))

def check_fingerprint(saved, current):
    if saved != current: raise RuntimeError('Code/config/data/reference/environment changed; use a new run ID')

def setup(run_id):
    if not run_id.replace('-','').replace('_','').isalnum(): raise ValueError('Invalid run ID')
    cp=ROOT/'configs/joint_contamination_v1.json'; config=read(cp)
    jp=ROOT/config['joint_config']; bp=ROOT/config['baseline_config']
    joint=read(jp); base=read(bp); data=load_dataset(joint['dataset'])
    ref=ROOT/'runs'/config['reference_run']; parent=read(ref/'manifest.json')['fingerprint']
    assert digest(jp)==parent['config'] and digest(bp)==parent['baseline'] and data.sha256==parent['data']
    for rel,h in parent['code'].items():
        if digest(ROOT/rel)!=h: raise RuntimeError('Parent implementation changed')
    packages={p:importlib.metadata.version(p) for p in parent['packages']}
    assert packages==parent['packages']
    paths=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'experiments').glob('*.py'))
    paths += [ROOT/'scripts'/name for name in ['run_baselines.py','run_feature_pilot.py','run_joint_pilot.py','run_contamination.py','check_joint_replay.py','audit_joint_pilot.py','run_joint_contamination.py','audit_joint_contamination.py']]
    fingerprint={'config':digest(cp),'joint':digest(jp),'baseline':digest(bp),'protocol':digest(ROOT/config['protocol']),
        'data':data.sha256,'splits':digest(ref/'outer_splits.json'),'packages':packages,
        'code':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'reference_manifest':digest(ref/'manifest.json'),
        'decisions':{str(f):digest(ref/f'fold-{f}'/'decisions.json') for f in config['outer_folds']}}
    run=ROOT/'runs'/run_id
    if (run/'manifest.json').exists(): check_fingerprint(read(run/'manifest.json')['fingerprint'],fingerprint)
    else:
        run.mkdir(parents=True,exist_ok=False)
        for p in paths:
            out=run/'code'/p.relative_to(ROOT); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,out)
        for p,name in [(cp,'config.json'),(jp,'joint_config.json'),(bp,'baseline_config.json'),(ROOT/config['protocol'],'protocol.md'),(ref/'outer_splits.json','outer_splits.json')]: shutil.copy2(p,run/name)
        for f in config['outer_folds']:
            out=run/'frozen_decisions'/f'fold-{f}.json'; out.parent.mkdir(exist_ok=True); shutil.copy2(ref/f'fold-{f}'/'decisions.json',out)
        save(run/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'fingerprint':fingerprint,'python':sys.version,'purpose':config['purpose']})
    # Reject missing or modified frozen copies too.
    for path,key in [('config.json','config'),('joint_config.json','joint'),('baseline_config.json','baseline'),('protocol.md','protocol'),('outer_splits.json','splits')]:
        assert digest(run/path)==fingerprint[key]
    for f,h in fingerprint['decisions'].items(): assert digest(run/'frozen_decisions'/f'fold-{f}.json')==h
    for rel,h in fingerprint['code'].items(): assert digest(run/'code'/rel)==h
    return run,config,joint,base,data,ref

def ensure(path,value):
    old=cached(path)
    if old is not None:
        if old != value: raise ValueError(f'Completed artifact mismatch: {path}')
        return old
    return commit(path,value)

def verify_clean(folder, ref, fold, choices):
    for method in sorted({c['method'] for c in choices.values()}):
        for name in ['selector.json',*[f'game-{b}.json' for b in range(4)]]:
            assert stable(cached(folder/'refit'/method/name))==stable(cached(ref/f'fold-{fold}'/'refit'/method/name)),name
    for arm in choices:
        new=cached(folder/'outer'/f'{arm}.json'); old=cached(ref/f'fold-{fold}'/'outer'/f'{arm}.json')
        for key in ['candidate','features','selected_n','train_n','test_n','actual_rho','cell_fraction','metrics']: assert new[key]==old[key],(arm,key)
        assert stable(cached(folder/new['model_path']))==stable(cached(ref/f'fold-{fold}'/old['model_path'])),arm
    ensure(folder/'clean_equivalence.json',{'status':'passed','fold':fold,'arms':len(choices),'comparison':'Exact substantive selectors, games, rows, variables, coefficients, probabilities and metrics'})

def execute_condition(run,config,joint,base,data,ref,fold,rate,draw,label):
    plan=read(run/'outer_splits.json')['folds'][fold]
    train=np.array(plan['train_positions']); test=np.array(plan['test_positions'])
    choices=frozen_choices(read(run/'frozen_decisions'/f'fold-{fold}.json'),joint)
    folder=run/f'fold-{fold}'/label
    if rate:
        gate=cached(run/f'fold-{fold}'/'rate-00-draw-0'/'clean_equivalence.json')
        if not gate or gate['status']!='passed': raise RuntimeError('Clean equivalence required first')
    seed=seed_for(config['seed_root'],data.name,fold,draw,'contamination')
    changed,events=perturbed_data(data,train,test,rate,seed,config['iqr_multiplier'])
    ensure(folder/'contamination.json',events)
    ensure(folder/'context.json',{'fold':fold,'rate':rate,'draw':draw,'train_ids':data.row_ids[train].tolist(),
        'test_ids':data.row_ids[test].tolist(),'clean_test_hash':frame_hash(data.X.iloc[test]),
        'perturbed_train_hash':frame_hash(changed.X.iloc[train]),'decisions_sha256':digest(run/'frozen_decisions'/f'fold-{fold}.json'),
        'contamination_sha256':digest(folder/'contamination.json'),'utility_validation_inherits_contamination':True})
    selectors={}
    for method in joint['methods']:
        if method not in {c['method'] for c in choices.values()}: continue
        spec=next(m for m in base['methods'] if m['name']==method)
        selectors[method]=build_selector(changed,train,f'{data.name}:outer{fold}:refit',spec,joint,base,folder/'refit'/method)
    for arm,candidate in choices.items():
        if (ROOT/'STOP_REQUESTED').exists(): raise SystemExit(75)
        selector=selectors[candidate['method']]; features=subset_for(selector,candidate)
        assert len(features)==candidate['k']
        mask=sum(1<<list(data.X.columns).index(f) for f in features)
        path=folder/'refit'/candidate['method']/'models'/f'mask-{mask}.json'
        model=fit_evaluate(changed,train,selector['selected_positions'],test,features,base,seed_for(base['seed_root'],data.name,fold,'classifier'),path)
        ensure(folder/'outer'/f'{arm}.json',{'arm':arm,'fold':fold,'rate':rate,'draw':draw,'candidate':candidate,
            'features':features,'selected_n':len(selector['selected_ids']),'train_n':len(train),'test_n':len(test),
            'actual_rho':selector['actual_rho'],'cell_fraction':selector['actual_rho']*len(features)/len(data.X.columns),
            'metrics':model['metrics'],'model_path':path.relative_to(folder).as_posix(),'model_sha256':digest(path),
            'decisions_sha256':digest(run/'frozen_decisions'/f'fold-{fold}.json')})
    if rate==0: verify_clean(folder,ref,fold,choices)
    files={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*.json') if not p.name.startswith('condition_complete')}
    ensure(folder/'condition_complete.json',{'status':'computed_pending_audit','sha256':files,'arms':8})
    print(f'CONDITION fold={fold} {label} saved; clean_equivalence={rate==0}',flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); ap.add_argument('--fold',type=int,choices=range(5),required=True)
    ap.add_argument('--max-new-conditions',type=int,default=7); ap.add_argument('--only',help='Condition label; clean condition must exist first')
    args=ap.parse_args(); run,config,joint,base,data,ref=setup(args.run_id)
    if args.fold>0:
        gate=ROOT/'qa'/f'{args.run_id}_audit_fold0.json'
        if not gate.exists() or read(gate).get('conditions')!=7 or read(gate).get('status')!='passed': raise RuntimeError('Complete fold-0 audit required first')
    count=0
    with threadpool_limits(limits=1):
        for rate,draw,label in conditions(config):
            if args.only and args.only!=label: continue
            if (ROOT/'STOP_REQUESTED').exists(): raise SystemExit(75)
            folder=run/f'fold-{args.fold}'/label
            done=cached(folder/'condition_complete.json')
            if done:
                for rel,h in done['sha256'].items():
                    if digest(folder/rel)!=h: raise ValueError(f'Changed completed condition {folder}/{rel}')
                print(f'SKIP {args.fold} {label}',flush=True); continue
            if count>=args.max_new_conditions: break
            execute_condition(run,config,joint,base,data,ref,args.fold,rate,draw,label); count+=1

if __name__=='__main__': main()
