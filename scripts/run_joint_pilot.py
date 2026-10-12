"""Resumable HTRU2 three-level integration pilot, one durable game/model per task."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.data import load_dataset, digest, seed_for
from src.contracts import assert_disjoint, FeatureSelection
from src.samplers import select_rows
from src.preprocessing import TabularEncoder
from src.evaluation import classification_metrics, group_metrics
from experiments.feature_utility import FeatureUtility
from experiments.feature_path import redundancy_matrix
from experiments.shapley import permutation_shapley, exact_shapley
from experiments.joint_selection import candidate_grid, paths_from_games, subset_for, aggregate_candidates, make_decisions
from scripts.run_baselines import write_json
from scripts.run_feature_pilot import serial


def read(path):
    return json.loads(path.read_text('utf-8'))


def save(path, obj):
    write_json(path, serial(obj))


def cached(path):
    marker = path.with_suffix('.complete.json')
    if marker.exists():
        if read(marker)['sha256'] != digest(path):
            raise ValueError(f'Completed artifact modified: {path}')
        return read(path)
    return None


def commit(path, obj):
    save(path, obj)
    save(path.with_suffix('.complete.json'), {'sha256': digest(path)})
    return read(path)


def build_selector(data, positions, partition_id, method, config, base, dest):
    """Only a training slice enters all feature computations; no validation input."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'selector.json'
    previous = cached(path)
    if previous is not None:
        if previous['training_ids'] != data.row_ids[positions].tolist() or previous['method'] != method:
            raise ValueError('Cached selector training context mismatch')
        for rel, sha in previous['game_sha256'].items():
            if digest(dest / rel) != sha: raise ValueError('Cached game changed')
        return previous
    start = time.perf_counter()
    part = data.partition(positions, partition_id)
    sample_seed = seed_for(config['seed_root'], partition_id, 'selector_sampler')
    selected_local, _, diagnostic = select_rows(data.X.iloc[positions], part, data.numeric,
        data.categorical, method['rho'], sample_seed, method['strategy'], base['sampler'])
    selected = positions[selected_local]
    R = redundancy_matrix(data.X.iloc[selected], data.numeric, data.categorical)
    geometry_seconds = time.perf_counter() - start
    games = []
    for b in range(config['B']):
        if (ROOT / 'STOP_REQUESTED').exists():
            raise SystemExit(75)
        game_path = dest / f'game-{b}.json'
        result = cached(game_path)
        if result is None:
            t = time.perf_counter()
            subsample_seed = seed_for(config['seed_root'], partition_id, b, 'feature_resample')
            local, _, _ = select_rows(data.X.iloc[positions], part, data.numeric, data.categorical,
                config['subsample_rho'], subsample_seed, 'stratified_random', base['sampler'])
            pool = positions[local]
            game_seed = seed_for(config['seed_root'], partition_id, b, 'utility')
            permutation_seed = seed_for(config['seed_root'], partition_id, method['name'], b, 'permutations')
            game = FeatureUtility(data.X.iloc[pool], data.partition(pool, f'{partition_id}:resample{b}'),
                data.numeric, data.categorical, method, base, game_seed, config['utility_folds'])
            approx = permutation_shapley(game.utility, game.features, config['M'], permutation_seed)
            permutation_fits = game.fit_count
            exact = exact_shapley(game.utility, game.features)
            result = {'resample': b, 'subsample_positions': pool, 'subsample_seed': subsample_seed,
                'game_seed': game_seed, 'permutation_seed': permutation_seed,
                'context': game.context, 'context_hash': game.context_hash,
                'features': game.features, 'permutation': approx, 'exact_phi': exact,
                'max_abs_error': float(np.max(np.abs(approx['phi'] - exact))),
                'coalitions': game.cache_records(), 'permutation_logistic_fits': permutation_fits,
                'logistic_fits': game.fit_count, 'constant_evaluations': game.constant_count,
                'seconds': time.perf_counter() - t}
            result = commit(game_path, result)
        games.append(result)
    paths = paths_from_games(tuple(data.X.columns), games, R, config)
    result = {'partition_id': partition_id, 'method': method, 'features': list(data.X.columns),
        'training_positions': positions, 'training_ids': data.row_ids[positions],
        'selected_positions': selected, 'selected_ids': data.row_ids[selected],
        'sample_seed': sample_seed, 'actual_rho': len(selected) / len(positions),
        'sampler_diagnostics': diagnostic, 'redundancy': R, 'paths': paths,
        'game_sha256': {f'game-{b}.json': digest(dest / f'game-{b}.json') for b in range(config['B'])},
        'geometry_seconds': geometry_seconds, 'game_seconds': sum(g['seconds'] for g in games),
        'selector_compute_seconds': geometry_seconds + sum(g['seconds'] for g in games),
        'logistic_fits': sum(g['logistic_fits'] for g in games),
        'constant_evaluations': sum(g['constant_evaluations'] for g in games)}
    result = commit(path, result)
    print(f'Selector {partition_id} {method["name"]}: fits={result["logistic_fits"]} seconds={result["selector_compute_seconds"]:.2f}', flush=True)
    return result


def fit_evaluate(data, train, selected, validation, features, base, model_seed, path):
    part = data.partition(train, 'model_train')
    assert_disjoint(part, data.row_ids[validation], data.group_ids[validation])
    FeatureSelection(tuple(features), part.partition_id).validate(part)
    selected = np.asarray(selected, int)
    if len(selected) != len(set(selected)) or not set(selected) <= set(train):
        raise ValueError('Selected rows outside training partition')
    context = {'training_ids': data.row_ids[train].tolist(), 'selected_ids': data.row_ids[selected].tolist(),
               'evaluation_ids': data.row_ids[validation].tolist(), 'features': list(features), 'model_seed': model_seed}
    old = cached(path)
    if old is not None:
        if old['context'] != context: raise ValueError('Model cache context mismatch')
        return old
    start = time.perf_counter()
    numeric = tuple(f for f in features if f in data.numeric)
    categorical = tuple(f for f in features if f in data.categorical)
    encoder = TabularEncoder(numeric, categorical)
    Z = encoder.fit_transform(data.X.iloc[selected][list(features)])
    params = {k: v for k, v in base['model'].items() if k != 'class'}
    model = LogisticRegression(**params, random_state=model_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model.fit(Z, data.y[selected])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Classifier did not converge')
    fit_seconds = time.perf_counter() - start
    t = time.perf_counter()
    probabilities = model.predict_proba(encoder.transform(data.X.iloc[validation][list(features)]))[:, list(model.classes_).index(1)]
    predict_seconds = time.perf_counter() - t
    return commit(path, {'context': context, 'selected_positions': selected, 'evaluation_positions': validation,
        'encoder': encoder.diagnostics(data.X.iloc[validation][list(features)]),
        'coef': model.coef_, 'intercept': model.intercept_, 'classes': model.classes_, 'n_iter': model.n_iter_,
        'probabilities': probabilities, 'targets': data.y[validation],
        'metrics': classification_metrics(data.y[validation], probabilities, base['threshold']),
        'group_metrics': group_metrics(data.audit.iloc[validation], data.y[validation], probabilities),
        'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds, 'logistic_fits': 1,
        'warnings': [{'type': w.category.__name__, 'message': str(w.message)} for w in caught]})


def evaluate_candidates(data, train, validation, selectors, config, base, model_seed, dest):
    rows = []
    for c in candidate_grid(config, len(data.X.columns)):
        selector = selectors[c['method']]
        features = subset_for(selector, c)
        mask = sum(1 << list(data.X.columns).index(f) for f in features)
        path = dest / c['method'] / 'models' / f'mask-{mask}.json'
        model = fit_evaluate(data, train, selector['selected_positions'], validation, features, base, model_seed, path)
        rows.append({**c, 'features': features, 'mask': mask, 'f1': model['metrics']['f1'],
                     'actual_rho': selector['actual_rho'], 'model_path': path.relative_to(dest).as_posix(),
                     'model_sha256': digest(path)})
    commit(dest / 'candidates.json', rows)
    return serial(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id', required=True)
    ap.add_argument('--fold', type=int); ap.add_argument('--max-contexts', type=int, default=80)
    ap.add_argument('--stop-after-inner', type=int)
    args = ap.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum() or args.max_contexts < 1:
        raise ValueError('Invalid run ID or budget')
    cp = ROOT / 'configs/joint_pilot_v1.json'; bp = ROOT / 'configs/baseline_pilot_v1.json'
    config = read(cp); base = read(bp); data = load_dataset(config['dataset'])
    if args.fold is not None and args.fold not in config['outer_folds']: raise ValueError('Invalid outer fold')
    if args.stop_after_inner is not None and args.stop_after_inner not in range(config['inner_folds']): raise ValueError('Invalid inner bound')
    parent = ROOT / 'runs' / config['parent_run'] / data.name / 'splits.json'
    splits = read(parent)
    if splits['data_sha256'] != data.sha256: raise ValueError('Parent data changed')
    paths = sorted((ROOT / 'src').glob('*.py')) + [ROOT / p for p in [
        'experiments/feature_utility.py', 'experiments/feature_path.py', 'experiments/shapley.py',
        'experiments/joint_selection.py', 'scripts/run_baselines.py', 'scripts/run_feature_pilot.py', 'scripts/run_joint_pilot.py']]
    fingerprint = {'config': digest(cp), 'baseline': digest(bp), 'data': data.sha256, 'splits': digest(parent),
        'code': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']}}
    run = ROOT / 'runs' / args.run_id
    if (run / 'manifest.json').exists():
        if read(run / 'manifest.json')['fingerprint'] != fingerprint:
            raise RuntimeError('Changed code/configuration; use a new run ID')
    else:
        run.mkdir(parents=True, exist_ok=False)
        for p in paths:
            out = run / 'code' / p.relative_to(ROOT); out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)
        shutil.copy2(cp, run / 'config.json'); shutil.copy2(bp, run / 'baseline_config.json'); shutil.copy2(parent, run / 'outer_splits.json')
        save(run / 'manifest.json', {'fingerprint': fingerprint, 'purpose': config['purpose'],
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'official_adult_test_used': False})
    completed_contexts = 0
    with threadpool_limits(limits=1):
        for fold in config['outer_folds']:
            if args.fold is not None and args.fold != fold: continue
            outer = splits['folds'][fold]; train = np.array(outer['train_positions']); test = np.array(outer['test_positions'])
            assert_disjoint(data.partition(train, f'outer{fold}'), data.row_ids[test], data.group_ids[test])
            fold_dir = run / f'fold-{fold}'
            inner_seed = seed_for(config['seed_root'], data.name, fold, 'inner_split')
            cv = StratifiedGroupKFold(n_splits=config['inner_folds'], shuffle=True, random_state=inner_seed)
            inner = [(train[tr], train[va]) for tr, va in cv.split(data.X.iloc[train], data.y[train], data.group_ids[train])]
            plan = {'seed': inner_seed, 'folds': [{'training_positions': tr.tolist(), 'validation_positions': va.tolist()} for tr, va in inner]}
            old_plan = cached(fold_dir / 'inner_splits.json')
            if old_plan is not None and old_plan != plan: raise ValueError('Inner folds changed')
            if old_plan is None: commit(fold_dir / 'inner_splits.json', plan)
            evaluations = []
            for i, (itr, iva) in enumerate(inner):
                assert_disjoint(data.partition(itr, 'inner'), data.row_ids[iva], data.group_ids[iva])
                dest = fold_dir / f'inner-{i}'; selectors = {}
                for method_name in config['methods']:
                    path = dest / method_name
                    if not (path / 'selector.complete.json').exists():
                        if completed_contexts >= args.max_contexts: print('Context budget reached', flush=True); return
                        completed_contexts += 1
                    method = next(m for m in base['methods'] if m['name'] == method_name)
                    selectors[method_name] = build_selector(data, itr, f'{data.name}:outer{fold}:inner{i}', method, config, base, path)
                evaluations.append(evaluate_candidates(data, itr, iva, selectors, config, base,
                    seed_for(config['seed_root'], data.name, fold, i, 'inner_model'), dest))
                print(f'Inner outer={fold} inner={i}: 128 candidates saved', flush=True)
                if args.stop_after_inner == i: return
            aggregated = aggregate_candidates(evaluations, config, len(data.X.columns))
            decisions = {**make_decisions(aggregated, config, len(data.X.columns)), 'candidate_scores': aggregated}
            decision_path = fold_dir / 'decisions.json'
            old = cached(decision_path)
            if old is not None and old != decisions: raise ValueError('Inner decision changed')
            if old is None: commit(decision_path, decisions)
            # Decisions are durable before any outer evaluation.
            selectors = {}
            for method_name in config['methods']:
                dest = fold_dir / 'refit' / method_name
                if not (dest / 'selector.complete.json').exists():
                    if completed_contexts >= args.max_contexts: print('Context budget reached before refit', flush=True); return
                    completed_contexts += 1
                method = next(m for m in base['methods'] if m['name'] == method_name)
                selectors[method_name] = build_selector(data, train, f'{data.name}:outer{fold}:refit', method, config, base, dest)
            for arm, candidate in decisions['selections'].items():
                selector = selectors[candidate['method']]; features = subset_for(selector, candidate)
                mask = sum(1 << list(data.X.columns).index(f) for f in features)
                dest = fold_dir / 'refit' / candidate['method'] / 'models' / f'mask-{mask}.json'
                model = fit_evaluate(data, train, selector['selected_positions'], test, features, base,
                    seed_for(base['seed_root'], data.name, fold, 'classifier'), dest)
                commit(fold_dir / 'outer' / f'{arm}.json', {'arm': arm, 'fold': fold, 'candidate': candidate,
                    'features': features, 'selected_n': len(selector['selected_ids']), 'train_n': len(train),
                    'test_n': len(test), 'actual_rho': selector['actual_rho'], 'cell_fraction': selector['actual_rho'] * len(features) / len(data.X.columns),
                    'metrics': model['metrics'], 'model_path': dest.relative_to(fold_dir).as_posix(),
                    'model_sha256': digest(dest), 'decisions_sha256': digest(decision_path)})
            print(f'OUTER fold={fold} completed: eight comparisons', flush=True)
    print('Requested joint pilot completed', flush=True)


if __name__ == '__main__': main()
