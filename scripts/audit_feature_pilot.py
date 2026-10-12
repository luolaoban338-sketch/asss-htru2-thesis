"""Independent coalition-table audit using all permutations, plus direct refits."""
from pathlib import Path
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data import load_dataset, digest, seed_for
from src.samplers import select_rows
from experiments.feature_utility import FeatureUtility
from experiments.shapley import corrected_scores
from scripts.run_baselines import write_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-id', required=True); args = ap.parse_args()
    run = ROOT / 'runs' / args.run_id
    config = json.loads((run / 'config.json').read_text('utf-8'))
    base = json.loads((run / 'baseline_config.json').read_text('utf-8'))
    manifest = json.loads((run / 'manifest.json').read_text('utf-8'))
    for p, sha in manifest['fingerprint']['code'].items(): assert digest(run / 'code' / p) == sha
    assert digest(run / 'config.json') == manifest['fingerprint']['config_sha256']
    assert digest(run / 'baseline_config.json') == manifest['fingerprint']['baseline_sha256']
    data = load_dataset(config['dataset'])
    parent_split_path = ROOT / 'runs' / config['parent_run'] / data.name / 'splits.json'
    assert digest(parent_split_path) == manifest['fingerprint']['split_sha256']
    parent_positions = np.array(json.loads(parent_split_path.read_text('utf-8'))['folds'][config['parent_training_fold']]['train_positions'])
    pool = set(manifest['training_pool_ids']); forbidden = set(manifest['excluded_outer_test_ids'])
    assert pool.isdisjoint(forbidden) and pool | forbidden == set(data.row_ids)
    game_rows = []; feature_rows = []; results = {}; exact_error = 0.; refits = 0
    with threadpool_limits(limits=1):
        for b in range(config['B']):
            shared_subsample = None
            expected_seed = seed_for(config['seed_root'], data.name, b, 'feature_resample')
            selected_local, _, _ = select_rows(data.X.iloc[parent_positions], data.partition(parent_positions, f'{data.name}:parent_train'),
                data.numeric, data.categorical, config['subsample_rho'], expected_seed, 'stratified_random', base['sampler'])
            expected_positions = parent_positions[selected_local]
            for name in config['methods']:
                dest = run / f'resample-{b}' / name
                record = json.loads((dest / 'record.json').read_text('utf-8'))
                assert record['status'] == 'completed' and digest(dest / 'game.json') == record['game_sha256']
                r = json.loads((dest / 'game.json').read_text('utf-8')); results[b, name] = r
                positions = np.array(r['subsample_positions'])
                np.testing.assert_array_equal(positions, expected_positions)
                assert r['subsample_seed'] == expected_seed
                assert r['game_seed'] == seed_for(config['seed_root'], data.name, b, 'feature_utility')
                assert r['permutation_seed'] == seed_for(config['seed_root'], data.name, b, name, 'permutations')
                assert len(set(positions)) == len(positions)
                assert set(data.row_ids[positions]) <= pool and not set(data.row_ids[positions]) & forbidden
                if shared_subsample is None: shared_subsample = positions
                else: np.testing.assert_array_equal(shared_subsample, positions)
                method = next(m for m in base['methods'] if m['name'] == name)
                game = FeatureUtility(data.X.iloc[positions], data.partition(positions, f'{data.name}:resample{b}'),
                    data.numeric, data.categorical, method, base, r['game_seed'], config['utility_folds'])
                assert game.context_hash == r['context_hash'] and json.loads(json.dumps(game.context)) == r['context']
                for f in game.folds:
                    assert set(game.ids[f['selected']]) <= pool
                    assert set(game.ids[f['validation']]) <= pool
                    assert not set(game.groups[f['selected']]) & set(game.groups[f['validation']])
                d = len(r['features']); coalitions = {c['mask']: c for c in r['coalitions']}
                assert len(coalitions) == 2 ** d and set(coalitions) == set(range(2 ** d))
                values = np.array([coalitions[m]['utility'] for m in range(2 ** d)])
                for mask, c in coalitions.items():
                    assert c['context_hash'] == r['context_hash']
                    assert c['features'] == [r['features'][j] for j in range(d) if mask & (1 << j)]
                    assert len(c['fold_f1']) == config['utility_folds']
                    assert c['utility'] == np.mean(c['fold_f1']) and 0 <= c['utility'] <= 1
                for subset in [(), (r['features'][0],), tuple(r['features'])]:
                    mask = sum(1 << r['features'].index(c) for c in subset)
                    assert game.utility(subset) == values[mask]
                refits += game.fit_count
                # Alternative reference: enumerate all d! orders, rather than subset weights.
                total = np.zeros(d); n = 0
                for order in itertools.permutations(range(d)):
                    mask = 0
                    for j in order:
                        newmask = mask | (1 << j); total[j] += values[newmask] - values[mask]; mask = newmask
                    n += 1
                enumerated = total / n
                error = float(np.max(np.abs(enumerated - np.array(r['exact_phi'])))); exact_error = max(error, exact_error)
                assert error < 1e-12
                permutations = r['permutation']['permutations']; marginal = np.zeros((config['M'], d))
                assert len(permutations) == config['M']
                for m, order in enumerate(permutations):
                    assert sorted(order) == list(range(d)); mask = 0
                    for j in order:
                        newmask = mask | (1 << j); marginal[m, j] = values[newmask] - values[mask]; mask = newmask
                np.testing.assert_array_equal(marginal, r['permutation']['marginals'])
                np.testing.assert_allclose(marginal.sum(axis=1), values[-1] - values[0], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(marginal.mean(axis=0), r['permutation']['phi'])
                assert r['max_abs_approximation_error'] == float(np.max(np.abs(marginal.mean(axis=0) - np.array(r['exact_phi']))))
                np.testing.assert_array_equal(marginal.var(axis=0, ddof=1) / config['M'], r['permutation']['mc_variance_of_mean'])
                assert r['total_logistic_fits'] == config['utility_folds'] * (2**d-1) and r['constant_evaluations'] == 3
                game_rows.append({'resample': b, 'method': name, 'exact_reference_error': error,
                    'max_abs_approximation_error': r['max_abs_approximation_error'],
                    'permutation_fits': r['permutation_phase_logistic_fits'], 'exact_inclusive_fits': r['total_logistic_fits'],
                    'empty_utility': values[0], 'full_utility': values[-1], 'seconds': r['seconds']})
    for name in config['methods']:
        rs = [results[b, name] for b in range(config['B'])]
        estimates = np.array([r['permutation']['phi'] for r in rs])
        mc = np.array([r['permutation']['mc_variance_of_mean'] for r in rs])
        exact = np.array([r['exact_phi'] for r in rs])
        corrected = corrected_scores(estimates, mc, config['lambda_diagnostic'])
        reference_sd = np.sqrt(np.maximum(0, estimates.var(axis=0, ddof=1) - mc.mean(axis=0)))
        np.testing.assert_array_equal(corrected['corrected_sd'], reference_sd)
        for j, feature in enumerate(rs[0]['features']):
            feature_rows.append({'method': name, 'feature': feature, 'mean_approx': corrected['mean'][j],
                'mean_exact': exact[:, j].mean(), 'observed_variance': corrected['observed_variance'][j],
                'mean_mc_variance': corrected['mean_mc_variance'][j], 'corrected_sd': corrected['corrected_sd'][j],
                'exact_across_resample_sd': exact[:, j].std(ddof=1), 'clipped_to_zero': bool(corrected['clipped_to_zero'][j]),
                'score_lambda_0_5': corrected['score'][j], 'mean_absolute_mc_error': np.abs(estimates[:, j] - exact[:, j]).mean()})
    games = pd.DataFrame(game_rows); features = pd.DataFrame(feature_rows)
    assert len(games) == config['planned_games'] and games.exact_inclusive_fits.sum() == config['planned_logistic_fits']
    games.to_csv(run / 'game_summary.csv', index=False); features.to_csv(run / 'feature_summary.csv', index=False)
    write_json(run / 'qa/integrity_checks.json', {'status': 'passed', 'games': len(games), 'exact_logistic_fits': int(games.exact_inclusive_fits.sum()),
        'independent_audit_refits': refits, 'all_permutation_exact_reference_max_error': exact_error,
        'outer_test_evaluated': False, 'audit_sha256': digest(Path(__file__)),
        'checks': ['code and data identities', 'parent training-only inclusion', 'shared subsamples across methods',
            'group split and frozen selection reconstruction', 'all coalition means', 'direct empty/singleton/full refits',
            'all d! permutations as an independent exact reference per game', 'MC marginals and mean variance', 'corrected variance formula']})
    print(games.to_string(index=False))
    print(f'Validated {len(games)} games and {refits} direct audit refits; max exact reference error {exact_error:.3g}')


if __name__ == '__main__': main()
