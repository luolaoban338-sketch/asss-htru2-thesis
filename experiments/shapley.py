"""Exact small-game reference, iid permutation estimator, variance correction."""
import math
import numpy as np


def exact_shapley(utility, features):
    features = tuple(features); d = len(features)
    if not 1 <= d <= 8 or len(set(features)) != d:
        raise ValueError('Exact validation supports 1 to 8 unique original variables')
    values = {mask: utility(tuple(features[j] for j in range(d) if mask & (1 << j))) for mask in range(1 << d)}
    phi = np.zeros(d)
    for j in range(d):
        for mask in range(1 << d):
            if mask & (1 << j): continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
            phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    if not np.isclose(phi.sum(), values[(1 << d) - 1] - values[0], atol=1e-12, rtol=0):
        raise ArithmeticError('Exact Shapley efficiency failed')
    return phi


def permutation_shapley(utility, features, M, seed):
    features = tuple(features); d = len(features)
    if d < 1 or len(set(features)) != d or type(M) is not int or M < 2:
        raise ValueError('Unique features and at least two iid permutations required')
    rng = np.random.default_rng(seed)
    permutations = np.array([rng.permutation(d) for _ in range(M)])
    marginal = np.zeros((M, d)); empty = utility(())
    for m, order in enumerate(permutations):
        subset = []; previous = empty
        for j in order:
            subset.append(features[j]); value = utility(tuple(subset))
            marginal[m, j] = value - previous; previous = value
    gap = utility(features) - empty
    np.testing.assert_allclose(marginal.sum(axis=1), gap, atol=1e-12, rtol=0)
    return {'phi': marginal.mean(axis=0), 'mc_variance_of_mean': marginal.var(axis=0, ddof=1) / M,
        'marginals': marginal, 'permutations': permutations, 'efficiency_gap': gap}


def corrected_scores(estimates, mc_variances, penalty):
    estimates = np.asarray(estimates, dtype=float); mc = np.asarray(mc_variances, dtype=float)
    if estimates.ndim != 2 or len(estimates) < 2 or mc.shape != estimates.shape:
        raise ValueError('At least two matching resampling estimates required')
    if not np.isfinite(estimates).all() or not np.isfinite(mc).all() or (mc < 0).any() or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid variance or penalty')
    mean = estimates.mean(axis=0); observed = estimates.var(axis=0, ddof=1); noise = mc.mean(axis=0)
    raw = observed - noise; corrected = np.maximum(0., raw)
    return {'mean': mean, 'observed_variance': observed, 'mean_mc_variance': noise,
        'raw_corrected_variance': raw, 'corrected_sd': np.sqrt(corrected),
        'clipped_to_zero': raw < 0, 'score': mean - penalty * np.sqrt(corrected)}
