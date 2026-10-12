"""Bounded full-selection resamples. No outer evaluation data enters the engine."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.metadata
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from src.data import Dataset, load_dataset, digest, seed_for
from src.samplers import select_rows
from src.contracts import assert_disjoint
from experiments.joint_selection import aggregate_candidates, make_decisions, subset_for
from scripts.run_joint_pilot import read, save, cached, commit, build_selector, evaluate_candidates


def restrict_to_pool(data, positions):
    positions = np.asarray(positions, int)
    if len(positions) != len(set(positions)) or np.any(positions < 0) or np.any(positions >= len(data.y)):
        raise ValueError('Invalid development pool')
    return Dataset(data.name, data.X.iloc[positions].reset_index(drop=True).copy(),
                   data.y[positions].copy(), data.row_ids[positions].copy(), data.group_ids[positions].copy(),
                   data.audit.iloc[positions].reset_index(drop=True).copy(), data.numeric, data.categorical, data.sha256)


def durable(path, obj):
    from scripts.run_feature_pilot import serial
    obj = serial(obj)
    old = cached(path)
    if old is not None:
        if old != obj:
            raise ValueError(f'Changed completed context: {path}')
        return old
    return commit(path, obj)


def prepare(run_id):
    if not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid run ID')
    cp = ROOT/'configs/stability_pilot_v1.json'; cfg = read(cp)
    jp = ROOT/cfg['joint_config']; joint = read(jp)
    bp = ROOT/'configs/baseline_pilot_v1.json'; base = read(bp)
    split_path = ROOT/'runs'/joint['parent_run']/joint['dataset']/'splits.json'
    splits = read(split_path); original = load_dataset(joint['dataset'])
    if original.sha256 != splits['data_sha256']:
        raise ValueError('Data/split mismatch')
    fold = splits['folds'][cfg['outer_fold']]
    positions = np.array(fold['train_positions']); excluded = np.array(fold['test_positions'])
    assert_disjoint(original.partition(positions, 'development_pool'), original.row_ids[excluded], original.group_ids[excluded])
    pool_record = {'original_positions': positions.tolist(), 'row_ids': original.row_ids[positions].tolist(),
                   'group_ids': original.group_ids[positions].tolist(),
                   'excluded_row_ids': original.row_ids[excluded].tolist(),
                   'excluded_group_ids': original.group_ids[excluded].tolist(), 'outer_fold': cfg['outer_fold']}
    data = restrict_to_pool(original, positions)
    paths = sorted((ROOT/'src').glob('*.py')) + [ROOT/p for p in [
        'experiments/feature_utility.py', 'experiments/feature_path.py', 'experiments/shapley.py',
        'experiments/joint_selection.py', 'experiments/stability.py', 'scripts/run_baselines.py',
        'scripts/run_feature_pilot.py', 'scripts/run_joint_pilot.py', 'scripts/run_stability_pilot.py']]
    protocol = ROOT/'docs/11_complete_selection_stability_protocol.md'
    fingerprint = {'config': digest(cp), 'joint_config': digest(jp), 'baseline': digest(bp),
                   'protocol': digest(protocol), 'splits': digest(split_path), 'data': data.sha256,
                   'code': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
                   'python': sys.version,
                   'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']}}
    run = ROOT/'runs'/run_id
    if (run/'manifest.json').exists():
        if read(run/'manifest.json')['fingerprint'] != fingerprint:
            raise RuntimeError('Changed implementation/configuration/environment; use a new run ID')
        for name, sha in [('config.json',fingerprint['config']), ('joint_config.json',fingerprint['joint_config']),
                          ('baseline_config.json',fingerprint['baseline']), ('outer_splits.json',fingerprint['splits']),
                          ('protocol.md',fingerprint['protocol'])]:
            if digest(run/name) != sha: raise ValueError(f'Changed snapshot: {name}')
        for rel, sha in fingerprint['code'].items():
            if digest(run/'code'/rel) != sha: raise ValueError(f'Changed code snapshot: {rel}')
    else:
        run.mkdir(parents=True, exist_ok=False)
        for p in paths:
            out = run/'code'/p.relative_to(ROOT); out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)
        for p, name in [(cp,'config.json'),(jp,'joint_config.json'),(bp,'baseline_config.json'),
                        (split_path,'outer_splits.json'),(protocol,'protocol.md')]:
            shutil.copy2(p, run/name)
        save(run/'manifest.json', {'fingerprint':fingerprint, 'purpose':cfg['purpose'],
             'created_at_utc':datetime.now(timezone.utc).isoformat(), 'outer_test_evaluated':False})
    durable(run/'development_pool.json', pool_record)
    return run, data, cfg, joint, base


def run_replicate(run, data, cfg, joint, base, replicate):
    if replicate not in range(cfg['replicates']): raise ValueError('Invalid replicate')
    folder = run/f'replicate-{replicate}'
    receipt = cached(folder/'result.json')
    if receipt is not None:
        for rel, sha in receipt['artifact_sha256'].items():
            if digest(folder/rel) != sha: raise ValueError(f'Changed artifact: {rel}')
        print(f'SKIP verified replicate={replicate}', flush=True)
        return receipt
    tick = time.perf_counter(); started = datetime.now(timezone.utc).isoformat()
    resumed = folder.exists() and any(folder.rglob('*.complete.json'))
    positions = np.arange(len(data.y))
    sample_seed = seed_for(cfg['seed_root'], data.name, replicate, 'complete_resample')
    chosen, _, _ = select_rows(data.X, data.partition(positions, 'development_pool'), data.numeric, data.categorical,
                              cfg['resample_rho'], sample_seed, 'stratified_random', base['sampler'])
    train = positions[chosen]
    durable(folder/'resample.json', {'replicate':replicate, 'seed':sample_seed, 'positions':train,
            'row_ids':data.row_ids[train], 'group_ids':data.group_ids[train], 'pool_n':len(data.y),
            'actual_rho':len(train)/len(data.y)})
    inner_seed = seed_for(cfg['seed_root'], data.name, replicate, 'inner_split')
    cv = StratifiedGroupKFold(n_splits=joint['inner_folds'], shuffle=True, random_state=inner_seed)
    inner = [(train[tr],train[va]) for tr,va in cv.split(data.X.iloc[train],data.y[train],data.group_ids[train])]
    durable(folder/'inner_splits.json', {'seed':inner_seed, 'folds':[
        {'training_positions':tr, 'validation_positions':va} for tr,va in inner]})
    # Seed changes are prespecified by replicate identity. All other joint settings stay fixed.
    config = {**joint, 'seed_root':cfg['seed_root']}
    evaluations = []
    for i, (tr,va) in enumerate(inner):
        assert_disjoint(data.partition(tr,'inner'),data.row_ids[va],data.group_ids[va])
        dest = folder/f'inner-{i}'; selectors = {}
        for name in config['methods']:
            method = next(m for m in base['methods'] if m['name'] == name)
            selectors[name] = build_selector(data,tr,f'{data.name}:stability{replicate}:inner{i}',method,config,base,dest/name)
        evaluations.append(evaluate_candidates(data,tr,va,selectors,config,base,
            seed_for(cfg['seed_root'],data.name,replicate,i,'inner_model'),dest))
    scores = aggregate_candidates(evaluations,config,len(data.X.columns))
    decisions = durable(folder/'decisions.json', {**make_decisions(scores,config,len(data.X.columns)), 'candidate_scores':scores})
    selectors = {}
    for name in config['methods']:
        method = next(m for m in base['methods'] if m['name'] == name)
        selectors[name] = build_selector(data,train,f'{data.name}:stability{replicate}:refit',method,config,base,folder/'refit'/name)
    selections = {}
    for arm,candidate in decisions['selections'].items():
        selector = selectors[candidate['method']]; features = subset_for(selector,candidate)
        selections[arm] = {'candidate':candidate, 'features':features,
             'selected_ids':selector['selected_ids'], 'selected_n':len(selector['selected_ids']),
             'resample_n':len(train), 'actual_rho':selector['actual_rho'],
             'cell_fraction':selector['actual_rho']*len(features)/len(data.X.columns)}
    artifacts = {p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*.json'))}
    selectors_saved = [read(p) for p in folder.rglob('selector.json')]
    models = [p for p in folder.rglob('mask-*.json') if not p.name.endswith('.complete.json')]
    result = commit(folder/'result.json', {'replicate':replicate, 'selections':selections,
        'artifact_sha256':artifacts, 'outer_test_evaluated':False,
        'counts':{'selectors':len(selectors_saved),'games':len(selectors_saved)*config['B'],
                  'contribution_fits':sum(s['logistic_fits'] for s in selectors_saved),
                  'constant_evaluations':sum(s['constant_evaluations'] for s in selectors_saved),
                  'inner_scoring_fits':len(models), 'final_classifier_fits':0},
        'timing':{'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
                  'wall_seconds':time.perf_counter()-tick,'resumed':resumed,
                  'scope':'run_replicate call, including checks, selection and I/O, excluding imports and run preparation'}})
    print(f'COMPLETE replicate={replicate} counts={result["counts"]}',flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id',required=True)
    ap.add_argument('--replicate',type=int,required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        run,data,cfg,joint,base = prepare(args.run_id)
        run_replicate(run,data,cfg,joint,base,args.replicate)


if __name__ == '__main__': main()
