"""Training-only selectors and deterministic decisions from inner scores."""
import numpy as np
from experiments.feature_path import forward_path
from experiments.shapley import corrected_scores


def candidate_grid(config, d):
    return [dict(candidate_id=i, method=m, lam=lam, eta=eta, k=k)
            for i, (m, lam, eta, k) in enumerate(
                (m, lam, eta, k) for m in config['methods']
                for lam in config['lambda_grid'] for eta in config['eta_grid']
                for k in range(1, d + 1))]


def paths_from_games(features, games, redundancy, config):
    estimates = np.array([g['permutation']['phi'] for g in games])
    variances = np.array([g['permutation']['mc_variance_of_mean'] for g in games])
    paths = []
    for lam in config['lambda_grid']:
        scores = corrected_scores(estimates, variances, lam)
        for eta in config['eta_grid']:
            paths.append({'lam': lam, 'eta': eta, 'scores': scores,
                          'path': forward_path(features, scores['score'], redundancy, eta)})
    return paths


def subset_for(selector, candidate):
    path = next(p for p in selector['paths'] if p['lam'] == candidate['lam'] and p['eta'] == candidate['eta'])
    selected = set(path['path'][candidate['k'] - 1]['features'])
    # Canonical model order makes the all-variable fallback exactly the baseline.
    return tuple(f for f in selector['features'] if f in selected)


def aggregate_candidates(evaluations, config, d):
    if len(evaluations) != config['inner_folds']:
        raise ValueError('All inner folds required before budget selection')
    grid = candidate_grid(config, d)
    result = []
    for c in grid:
        folds = []
        for rows in evaluations:
            matches = [r for r in rows if r['candidate_id'] == c['candidate_id']]
            if len(matches) != 1 or any(matches[0][k] != v for k, v in c.items()):
                raise ValueError('Candidate grid changed or incomplete')
            folds.append(matches[0])
        f1 = np.array([r['f1'] for r in folds], float)
        rho = np.array([r['actual_rho'] for r in folds], float)
        if not np.isfinite(f1).all() or not np.isfinite(rho).all() or np.any((f1 < 0) | (f1 > 1) | (rho <= 0) | (rho > 1)):
            raise ValueError('Invalid fold metrics or row fractions')
        result.append({**c, 'fold_f1': f1.tolist(), 'mean_f1': float(f1.mean()),
                       'mean_actual_rho': float(rho.mean()), 'cell_fraction': float(rho.mean() * c['k'] / d)})
    return result


def choose_candidate(rows, threshold, tolerance=1e-12):
    if not np.isfinite(threshold) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid threshold/tolerance')
    if any(not np.isfinite(r['mean_f1']) or not np.isfinite(r['cell_fraction']) for r in rows):
        raise ValueError('Nonfinite candidate')
    eligible = [r for r in rows if r['mean_f1'] >= threshold - tolerance]
    if not eligible:
        raise ValueError('No feasible candidate; do not silently relax the tolerance')
    best = min(r['cell_fraction'] for r in eligible)
    eligible = [r for r in eligible if r['cell_fraction'] <= best + tolerance]
    best_k = min(r['k'] for r in eligible)
    eligible = [r for r in eligible if r['k'] == best_k]
    best_f1 = max(r['mean_f1'] for r in eligible)
    eligible = [r for r in eligible if r['mean_f1'] >= best_f1 - tolerance]
    return min(eligible, key=lambda r: r['candidate_id'])


def make_decisions(rows, config, d):
    tol = config['numeric_tolerance']
    full = [r for r in rows if r['method'] == 'full' and r['k'] == d]
    if not full or max(r['mean_f1'] for r in full) - min(r['mean_f1'] for r in full) > tol:
        raise ValueError('Full fallback is missing or inconsistent')
    threshold = full[0]['mean_f1'] - config['epsilon']
    choose = lambda candidates: choose_candidate(candidates, threshold, tol)
    sample = choose([r for r in rows if r['k'] == d])
    selections = {'full': choose(full), 'sample_only': sample,
        'feature_only': choose([r for r in rows if r['method'] == 'full']),
        'joint': choose(rows),
        'sequential': choose([r for r in rows if r['method'] == sample['method']])}
    for name, method in [('random', 'random_half'), ('ordinary', 'ordinary_half'), ('robust', 'robust_half')]:
        selections['joint_' + name] = choose([r for r in rows if r['method'] in ('full', method)])
    return {'baseline_inner_f1': full[0]['mean_f1'], 'threshold': threshold,
            'epsilon': config['epsilon'], 'selections': selections,
            'feasible_candidate_ids': [r['candidate_id'] for r in rows if r['mean_f1'] >= threshold - tol]}
