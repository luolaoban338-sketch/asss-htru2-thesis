"""Descriptive feature-set stability; no population-inference claims."""
from itertools import combinations
import numpy as np


def selection_matrix(feature_sets, features):
    features = tuple(features)
    if not features or len(set(features)) != len(features):
        raise ValueError('Feature universe must be nonempty and unique')
    rows = []
    for values in feature_sets:
        values = tuple(values)
        if len(set(values)) != len(values) or not set(values) <= set(features):
            raise ValueError('Unknown or duplicate selected features')
        rows.append([int(f in values) for f in features])
    if len(rows) < 2:
        raise ValueError('At least two complete selection runs required')
    return np.asarray(rows, dtype=int)


def stability_summary(feature_sets, features):
    """Nogueira et al. (2018), Eq. 2, plus all unordered Jaccard pairs.

    The zero-denominator cases return None, not an invented perfect score.
    Empty/empty Jaccard is defined as 1 by this project's explicit convention.
    """
    Z = selection_matrix(feature_sets, features)
    n, d = Z.shape
    frequencies = Z.mean(axis=0)
    sizes = Z.sum(axis=1)
    mean_fraction = float(sizes.mean() / d)
    denominator = mean_fraction * (1 - mean_fraction)
    numerator = float(Z.var(axis=0, ddof=1).mean())
    degenerate = 'all_empty' if not Z.any() else 'all_full' if Z.all() else None
    value = None if denominator == 0 else float(1 - numerator / denominator)
    pairs = []
    for i, j in combinations(range(n), 2):
        union = int(np.logical_or(Z[i], Z[j]).sum())
        intersection = int(np.logical_and(Z[i], Z[j]).sum())
        pairs.append({'i': i, 'j': j, 'intersection': intersection, 'union': union,
                      'jaccard': intersection / union if union else 1.0})
    return {'n_runs': n, 'd': d, 'selection_matrix': Z.tolist(),
            'frequencies': dict(zip(features, frequencies.tolist())),
            'sizes': sizes.tolist(), 'mean_k': float(sizes.mean()),
            'min_k': int(sizes.min()), 'max_k': int(sizes.max()),
            'nogueira': value, 'nogueira_status': degenerate or 'defined',
            'mean_unbiased_variance': numerator, 'chance_denominator': denominator,
            'pair_count': len(pairs), 'jaccard_mean': float(np.mean([p['jaccard'] for p in pairs])),
            'jaccard_pairs': pairs,
            'empty_union_convention': 'Jaccard(empty, empty)=1',
            'inference': 'descriptive; overlapping training resamples; pairs are not independent replicates'}
