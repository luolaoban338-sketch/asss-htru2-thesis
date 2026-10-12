"""Bounded contamination experiment; saved baseline splits and fixed methods."""
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
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.data import load_dataset, digest, seed_for
from src.contracts import assert_disjoint
from src.samplers import select_rows
from src.preprocessing import TabularEncoder
from src.evaluation import classification_metrics, group_metrics
from experiments.contamination import contaminate
from scripts.run_baselines import write_json


def conditions(config):
    for rate in config['rates']:
        for draw in ([config['clean_draw']] if rate == 0 else config['draws']):
            yield rate, draw, f'rate-{int(round(rate * 100)):02d}-draw-{draw}'


def setup_run(run_id):
    if not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid run ID')
    cp = ROOT / 'configs/contamination_pilot_v1.json'
    config = json.loads(cp.read_text('utf-8'))
    bp = ROOT / config['baseline_config']
    base = json.loads(bp.read_text('utf-8'))
    ref = ROOT / 'runs' / config['baseline_reference_run']
    reference_manifest = json.loads((ref / 'manifest.json').read_text('utf-8'))
    if reference_manifest['config_sha256'] != digest(bp):
        raise ValueError('Baseline configuration changed')
    code_paths = sorted((ROOT / 'src').glob('*.py')) + [
        ROOT / 'experiments/contamination.py', ROOT / 'scripts/run_baselines.py', Path(__file__).resolve()]
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths}
    for path, sha in reference_manifest['code_sha256'].items():
        if path.startswith('src/') and code.get(path) != sha:
            raise ValueError('Baseline implementation changed')
    packages = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']}
    if packages != reference_manifest['packages']:
        raise ValueError('Baseline package environment changed')
    fingerprint = {'code_sha256': code, 'config_sha256': digest(cp),
                   'baseline_config_sha256': digest(bp), 'packages': packages,
                   'reference_manifest_sha256': digest(ref / 'manifest.json'),
                   'split_sha256': {name: digest(ref / name / 'splits.json') for name in base['datasets']}}
    run = ROOT / 'runs' / run_id
    if (run / 'manifest.json').exists():
        old = json.loads((run / 'manifest.json').read_text('utf-8'))
        if old['fingerprint'] != fingerprint:
            raise RuntimeError('Implementation/config/environment changed; use a new run ID')
    else:
        run.mkdir(parents=True, exist_ok=False)
        for p in code_paths:
            target = run / 'code' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        shutil.copy2(cp, run / 'config.json')
        shutil.copy2(bp, run / 'baseline_config.json')
        write_json(run / 'manifest.json', {'fingerprint': fingerprint, 'purpose': config['purpose'],
                   'created_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                   'platform': platform.platform(), 'planned_fits': config['planned_fits'],
                   'official_adult_test_used': False})
    return run, config, base, ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dataset', choices=['htru2'], required=True)
    parser.add_argument('--max-jobs', type=int, default=140)
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    if args.max_jobs < 1:
        raise ValueError('Positive job limit required')
    run, config, base, ref = setup_run(args.run_id)
    data = load_dataset(args.dataset)
    splitpath = ref / data.name / 'splits.json'
    plan = json.loads(splitpath.read_text('utf-8'))
    if plan['data_sha256'] != data.sha256:
        raise ValueError('Dataset does not match baseline')
    write_json(run / data.name / 'splits.json', plan)
    jobs = skipped = 0
    for f in plan['folds']:
        fold = f['fold']
        if args.fold is not None and fold != args.fold:
            continue
        tr = np.array(f['train_positions']); te = np.array(f['test_positions'])
        train = data.partition(tr, f'{data.name}:outer{fold}')
        assert_disjoint(train, data.row_ids[te], data.group_ids[te])
        clean = data.X.iloc[tr].copy()
        for rate, draw, condition in conditions(config):
            seed = seed_for(config['seed_root'], data.name, fold, draw, 'contamination')
            X, pollution = contaminate(clean, data.row_ids[tr], data.numeric, rate, seed, config['iqr_multiplier'])
            condition_dir = run / data.name / f'fold-{fold}' / condition
            pp = condition_dir / 'contamination.json'
            if pp.exists() and json.loads(pp.read_text('utf-8')) != pollution:
                raise ValueError('Contamination plan changed')
            if not pp.exists():
                write_json(pp, pollution)
            pollution_hash = digest(pp)
            polluted_ids = {e['row_id'] for e in pollution['events']}
            for method in base['methods']:
                dest = condition_dir / method['name']
                record_path = dest / 'record.json'
                if record_path.exists():
                    old = json.loads(record_path.read_text('utf-8'))
                    if old['status'] != 'completed':
                        raise RuntimeError(f'Review previous failure: {dest}')
                    if old['contamination_sha256'] != pollution_hash:
                        raise ValueError('Saved contamination changed')
                    for filename, sha in old['artifact_sha256'].items():
                        if digest(dest / filename) != sha:
                            raise ValueError('Completed artifact changed')
                    skipped += 1
                    continue
                if jobs >= args.max_jobs:
                    print(f'Job bound reached; completed={jobs}, skipped={skipped}', flush=True)
                    return
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    sample_seed = seed_for(base['seed_root'], data.name, fold, 'sampler')
                    model_seed = seed_for(base['seed_root'], data.name, fold, 'classifier')
                    t0 = time.perf_counter()
                    with threadpool_limits(limits=base['thread_limit']):
                        t = time.perf_counter()
                        local, selected, diagnostic = select_rows(X, train, data.numeric, data.categorical,
                            method['rho'], sample_seed, method['strategy'], base['sampler'])
                        sampler_seconds = time.perf_counter() - t
                        selected_positions = tr[local]
                        t = time.perf_counter()
                        enc = TabularEncoder(data.numeric, data.categorical)
                        Z = enc.fit_transform(X.iloc[local])
                        params = {k: v for k, v in base['model'].items() if k != 'class'}
                        model = LogisticRegression(**params, random_state=model_seed)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            model.fit(Z, data.y[selected_positions])
                        warning_list = [{'type': w.category.__name__, 'message': str(w.message)} for w in caught]
                        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                            raise RuntimeError('Classifier convergence warning')
                        fit_seconds = time.perf_counter() - t
                        t = time.perf_counter()
                        probs = model.predict_proba(enc.transform(data.X.iloc[te]))[:, list(model.classes_).index(1)]
                        predict_seconds = time.perf_counter() - t
                    total_seconds = time.perf_counter() - t0
                    metrics = classification_metrics(data.y[te], probs, base['threshold'])
                    pd.DataFrame({'row_id': data.row_ids[te], 'group_id': data.group_ids[te],
                        'target': data.y[te], 'probability': probs, 'prediction': (probs >= base['threshold']).astype(int)
                        }).to_csv(dest / 'predictions.csv', index=False, float_format='%.17g')
                    write_json(dest / 'selection.json', {'selected_row_ids': selected.row_ids,
                        'selected_positions': selected_positions.tolist(), 'selected_local_positions': local.tolist(),
                        'original_features': list(data.X.columns), 'class_counts': selected.class_counts,
                        'requested_rho': method['rho'], 'actual_rho': len(local) / len(tr),
                        'sampler_diagnostics': diagnostic})
                    write_json(dest / 'encoder.json', enc.diagnostics(data.X.iloc[te]))
                    write_json(dest / 'model.json', {'type': 'LogisticRegression', 'parameters': model.get_params(),
                        'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                        'classes': model.classes_.tolist(), 'n_iter': model.n_iter_.tolist(), 'fit_n_rows': len(local)})
                    write_json(dest / 'group_metrics.json', group_metrics(data.audit.iloc[te], data.y[te], probs, base['threshold']))
                    retained = len(polluted_ids & set(selected.row_ids))
                    write_json(record_path, {'status': 'completed', 'dataset': data.name, 'fold': fold,
                        'method': method['name'], 'rate': rate, 'draw': draw, 'metrics': metrics,
                        'train_n': len(tr), 'test_n': len(te), 'selected_n': len(local),
                        'actual_rho': len(local) / len(tr), 'original_d': len(data.X.columns), 'encoded_d': Z.shape[1],
                        'contaminated_n': len(polluted_ids), 'retained_contaminated_n': retained,
                        'contamination_retention': retained / len(polluted_ids) if polluted_ids else None,
                        'contamination_fraction_selected': retained / len(local),
                        'contamination_sha256': pollution_hash,
                        'seeds': {'sampler': sample_seed, 'classifier': model_seed, 'contamination': seed},
                        'sampler_seconds': sampler_seconds, 'fit_seconds': fit_seconds,
                        'predict_seconds': predict_seconds, 'total_seconds': total_seconds,
                        'time_boundary': 'sampling through prediction; excludes contamination generation, IO and split loading',
                        'warnings': warning_list, 'artifact_sha256': {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
                        'completed_at_utc': datetime.now(timezone.utc).isoformat()})
                    jobs += 1
                    print(f'{data.name} fold={fold} {condition} {method["name"]} F1={metrics["f1"]:.5f} contaminated_kept={retained}/{len(polluted_ids)}', flush=True)
                except Exception as exc:
                    write_json(record_path, {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
                    raise
    print(f'Done {data.name}: completed={jobs}, skipped={skipped}', flush=True)


if __name__ == '__main__': main()
