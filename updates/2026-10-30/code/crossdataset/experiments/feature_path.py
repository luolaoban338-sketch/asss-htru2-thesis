"""Training-only association proxy and complete deterministic feature paths."""
import numpy as np
import pandas as pd
from src.preprocessing import TabularEncoder


def redundancy_matrix(selected_X, numeric, categorical):
    """Max absolute Spearman correlation across each original-variable block."""
    if len(selected_X) < 2: raise ValueError('At least two selected training rows required')
    encoder = TabularEncoder(numeric, categorical)
    Z = encoder.fit_transform(selected_X)
    ranks = pd.DataFrame(Z).rank(method='average').to_numpy(copy=True)
    ranks -= ranks.mean(axis=0)
    norms = np.sqrt((ranks ** 2).sum(axis=0))
    denominator = np.outer(norms, norms)
    corr = np.divide(ranks.T @ ranks, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    corr = np.clip(np.abs(corr), 0, 1)
    features = tuple(selected_X.columns); d = len(features); redundancy = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            block = corr[np.ix_(encoder.blocks_[features[i]], encoder.blocks_[features[j]])]
            redundancy[i, j] = redundancy[j, i] = float(block.max())
    return redundancy


def forward_path(features, scores, redundancy, eta):
    """G(S)=sum scores - eta/d * sum_{i<j in S} R[i,j]; keep every k."""
    features = tuple(features); scores = np.asarray(scores, dtype=float); R = np.asarray(redundancy, dtype=float)
    d = len(features)
    if not d or len(set(features)) != d or scores.shape != (d,) or R.shape != (d, d):
        raise ValueError('Unique original variables and matching dimensions required')
    if not np.isfinite(scores).all() or not np.isfinite(R).all() or not np.isfinite(eta) or eta < 0:
        raise ValueError('Finite scores, associations and nonnegative eta required')
    if np.any((R < 0) | (R > 1)) or not np.allclose(R, R.T, atol=1e-12, rtol=0) or np.any(np.diag(R) != 0):
        raise ValueError('Association matrix must be symmetric, in [0,1], with zero diagonal')
    remaining = list(range(d)); chosen = []; path = []; objective = 0.
    while remaining:
        increments = {j: float(scores[j] - eta / d * R[j, chosen].sum()) for j in remaining}
        # Original feature index resolves exact ties. Negative increments do not stop the path.
        pick = min(remaining, key=lambda j: (-increments[j], j))
        chosen.append(pick); remaining.remove(pick); objective += increments[pick]
        path.append({'k': len(chosen), 'features': [features[j] for j in chosen],
            'added': features[pick], 'increment': increments[pick], 'objective': objective})
    return path
