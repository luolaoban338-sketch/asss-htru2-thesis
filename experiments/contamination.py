"""Numeric contamination operates on a copy of a training partition only."""
import math
import numpy as np
import pandas as pd
from src.data import seed_for


def contaminate(X, row_ids, numeric, rate, seed, multiplier=10.0):
    if not 0 <= rate <= 1 or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Invalid contamination settings')
    ids = np.asarray(row_ids, dtype=str)
    if len(ids) != len(X) or len(set(ids)) != len(ids) or not len(X):
        raise ValueError('Invalid row identity')
    if not set(numeric) <= set(X.columns):
        raise ValueError('Numeric schema mismatch')
    out = X.copy(deep=True)
    scales = {}
    for col in numeric:
        values = pd.to_numeric(X[col], errors='raise').to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('Infinite input')
        observed = values[np.isfinite(values)]
        iqr = float(np.diff(np.quantile(observed, [.25, .75], method='linear'))[0]) if len(observed) else 0.0
        if iqr > 0:
            scales[col] = iqr
    count = math.ceil(rate * len(X))
    if count and not scales:
        raise ValueError('No positive-IQR numeric column')
    # Sorting identities makes draws stable under a reordered training frame.
    order = np.argsort(ids, kind='stable')
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(order)[:count]
    events = []
    for pos in chosen:
        eligible = [c for c in scales if pd.notna(X.iloc[pos][c])]
        if not eligible:
            raise ValueError(f'No observed eligible numeric cell in row {ids[pos]}')
        cell_rng = np.random.default_rng(seed_for(seed, 'cell', ids[pos]))
        col = eligible[int(cell_rng.integers(len(eligible)))]
        sign = int(cell_rng.choice([-1, 1]))
        before = float(X.iloc[pos][col])
        after = before + sign * multiplier * scales[col]
        if not np.isfinite(after) or after == before:
            raise ValueError('Contamination did not yield a finite changed value')
        out[col] = out[col].astype(float)
        out.iloc[pos, out.columns.get_loc(col)] = after
        events.append({'local_position': int(pos), 'row_id': str(ids[pos]), 'feature': col,
                       'before': before, 'after': after, 'sign': sign, 'iqr': scales[col]})
    return out, {'rate': float(rate), 'actual_rate': count / len(X), 'count': count,
                 'seed': int(seed), 'multiplier': multiplier, 'eligible_iqr': scales, 'events': events}
