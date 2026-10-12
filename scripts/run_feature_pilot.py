"""Bounded, resumable exact-reference development pilot on HTRU2 training only."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_dataset, digest, seed_for
from src.samplers import select_rows
from experiments.feature_utility import FeatureUtility
from experiments.shapley import exact_shapley, permutation_shapley, corrected_scores
from scripts.run_baselines import write_json


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id', required=True)
    ap.add_argument('--max-games', type=int, default=4); ap.add_argument('--resample', type=int)
    args = ap.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum() or args.max_games < 1:
        raise ValueError('Invalid run ID or game budget')
    cp = ROOT / 'configs/feature_pilot_v1.json'; bp = ROOT / 'configs/baseline_pilot_v1.json'
    config = json.loads(cp.read_text('utf-8')); base = json.loads(bp.read_text('utf-8'))
    if args.resample is not None and args.resample not in range(config['B']): raise ValueError('Invalid resample')
    data = load_dataset(config['dataset'])
    parent = ROOT / 'runs' / config['parent_run']
    split_path = parent / data.name / 'splits.json'
    split = json.loads(split_path.read_text('utf-8'))
    assert split['data_sha256'] == data.sha256
    parent_fold = split['folds'][config['parent_training_fold']]
    pool = np.array(parent_fold['train_positions']); heldout = np.array(parent_fold['test_positions'])
    assert not set(data.group_ids[pool]) & set(data.group_ids[heldout])
    parent_manifest = json.loads((parent / 'manifest.json').read_text('utf-8'))
    assert digest(bp) == parent_manifest['config_sha256']
    paths = sorted((ROOT / 'src').glob('*.py')) + [ROOT / 'experiments/feature_utility.py',
        ROOT / 'experiments/shapley.py', ROOT / 'scripts/run_baselines.py', Path(__file__).resolve()]
    fingerprint = {'config_sha256': digest(cp), 'baseline_sha256': digest(bp), 'data_sha256': data.sha256,
        'split_sha256': digest(split_path), 'code': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']}}
    run = ROOT / 'runs' / args.run_id
    if (run / 'manifest.json').exists():
        if json.loads((run / 'manifest.json').read_text('utf-8'))['fingerprint'] != fingerprint:
            raise RuntimeError('Changed implementation or configuration; use new run ID')
    else:
        run.mkdir(parents=True, exist_ok=False)
        for p in paths:
            out = run / 'code' / p.relative_to(ROOT); out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)
        shutil.copy2(cp, run / 'config.json'); shutil.copy2(bp, run / 'baseline_config.json')
        write_json(run / 'manifest.json', {'fingerprint': fingerprint, 'purpose': config['purpose'],
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'training_pool_ids': data.row_ids[pool].tolist(),
            'excluded_outer_test_ids': data.row_ids[heldout].tolist(), 'outer_test_evaluated': False})
    jobs = skipped = 0
    with threadpool_limits(limits=1):
        for b in range(config['B']):
            if args.resample is not None and args.resample != b: continue
            subsample_seed = seed_for(config['seed_root'], data.name, b, 'feature_resample')
            part = data.partition(pool, f'{data.name}:parent_train')
            local, _, _ = select_rows(data.X.iloc[pool], part, data.numeric, data.categorical,
                config['subsample_rho'], subsample_seed, 'stratified_random', base['sampler'])
            positions = pool[local]
            game_seed = seed_for(config['seed_root'], data.name, b, 'feature_utility')
            for name in config['methods']:
                dest = run / f'resample-{b}' / name
                if (dest / 'record.json').exists():
                    old = json.loads((dest / 'record.json').read_text('utf-8'))
                    if old['status'] != 'completed': raise RuntimeError('Review failed game before resuming')
                    assert digest(dest / 'game.json') == old['game_sha256']
                    skipped += 1; continue
                if jobs >= args.max_games:
                    print(f'Game bound reached: completed={jobs}, skipped={skipped}', flush=True); return
                dest.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
                method = next(m for m in base['methods'] if m['name'] == name)
                game = FeatureUtility(data.X.iloc[positions], data.partition(positions, f'{data.name}:resample{b}'),
                    data.numeric, data.categorical, method, base, game_seed, config['utility_folds'])
                permutation_seed = seed_for(config['seed_root'], data.name, b, name, 'permutations')
                approx = permutation_shapley(game.utility, game.features, config['M'], permutation_seed)
                pilot_fits = game.fit_count; pilot_utilities = len(game.cache)
                exact = exact_shapley(game.utility, game.features)
                assert game.fit_count == (2 ** len(game.features) - 1) * config['utility_folds']
                result = {'dataset': data.name, 'resample': b, 'method': name, 'features': game.features,
                    'subsample_positions': positions, 'subsample_seed': subsample_seed,
                    'game_seed': game_seed, 'permutation_seed': permutation_seed,
                    'context': game.context, 'context_hash': game.context_hash,
                    'permutation': approx, 'exact_phi': exact,
                    'max_abs_approximation_error': float(np.max(np.abs(approx['phi'] - exact))),
                    'coalitions': game.cache_records(), 'permutation_phase_logistic_fits': pilot_fits,
                    'permutation_phase_unique_utilities': pilot_utilities, 'total_logistic_fits': game.fit_count,
                    'constant_evaluations': game.constant_count, 'cache_hits': game.cache_hits,
                    'seconds': time.perf_counter() - start, 'outer_test_evaluated': False}
                write_json(dest / 'game.json', serial(result))
                write_json(dest / 'record.json', {'status': 'completed', 'game_sha256': digest(dest / 'game.json'),
                    'completed_at_utc': datetime.now(timezone.utc).isoformat()})
                jobs += 1
                print(f'{name} B={b} fits={game.fit_count} permutation_fits={pilot_fits} max_error={result["max_abs_approximation_error"]:.5f} seconds={result["seconds"]:.2f}', flush=True)
    print(f'Done: completed={jobs}, skipped={skipped}', flush=True)


if __name__ == '__main__': main()
