"""Training-only A3 candidate and explicit estimator-attempt boundary.

No command-line runner is provided. A3.1 must additionally provide verified
source/split loading, durable execution gates, watchdog and immutable outputs.
This module accepts only training pools during selection. Assertions cannot
prove caller-supplied row labels are truthful: the source loader must bind them.
"""
from dataclasses import dataclass
import math
import time
import warnings
import numpy as np
from .design import C_GRID, METHODS, seed
from .shapley_bound import corrected_scores, permutation_shapley


@dataclass(frozen=True)
class TrainingPool:
    X: np.ndarray
    y: np.ndarray
    row_ids: tuple
    features: tuple
    forbidden_ids: tuple = ()
    role: str = 'train'

    def __post_init__(self):
        X, y = np.array(self.X, dtype=float, copy=True), np.array(self.y, copy=True)
        ids, features = tuple(self.row_ids), tuple(self.features)
        if self.role != 'train' or X.ndim != 2 or y.shape != (len(X),):
            raise ValueError('Training-only aligned two-dimensional pool required')
        if X.shape[1] != len(features) or len(set(features)) != len(features) or not features:
            raise ValueError('Unique original features required')
        if len(ids) != len(X) or len(set(ids)) != len(ids) or set(ids) & set(self.forbidden_ids):
            raise ValueError('Invalid row identity or train/held-out leakage')
        if set(y.tolist()) != {0, 1} or not np.isfinite(X).all():
            raise ValueError('HTRU2 numeric finite data and both binary classes required')
        X.setflags(write=False); y.setflags(write=False)
        object.__setattr__(self, 'X', X); object.__setattr__(self, 'y', y)
        object.__setattr__(self, 'row_ids', ids); object.__setattr__(self, 'features', features)
        object.__setattr__(self, 'forbidden_ids', tuple(self.forbidden_ids))

    def take(self, indices, forbidden=()):
        idx = np.asarray(indices, dtype=int)
        return TrainingPool(self.X[idx], self.y[idx], tuple(self.row_ids[i] for i in idx),
                            self.features, tuple(set(self.forbidden_ids) | set(forbidden)))


def stratified_folds(pool, split_seed, n_splits=3):
    """Frozen inner splitter: independent class permutations, contiguous chunks.

    HTRU2 has singleton fingerprint groups. The bound loader must reject any
    group duplicates; group data with repeats is outside this candidate contract.
    """
    rng = np.random.default_rng(split_seed)
    bins = [[] for _ in range(n_splits)]
    for c in (0, 1):
        idx = np.flatnonzero(pool.y == c)
        if len(idx) < n_splits:
            raise ValueError('Each class must cover every validation fold')
        idx = idx[np.argsort(np.asarray(pool.row_ids)[idx], kind='stable')]
        for j, chunk in enumerate(np.array_split(rng.permutation(idx), n_splits)):
            bins[j].extend(chunk.tolist())
    result = []
    for values in bins:
        va = np.array(sorted(values), dtype=int)
        tr = np.setdiff1d(np.arange(len(pool.X)), va)
        result.append((tr, va))
    return result


def contribution_pool(pool, outer_fold, b, fraction=.8):
    # Historical class_quotas: ceil target, Hamilton remainder, no rescue.
    n, target = len(pool.X), math.ceil(fraction * len(pool.X))
    counts = [int(np.sum(pool.y == c)) for c in (0, 1)]
    q = [target * count // n for count in counts]
    order = sorted((0, 1), key=lambda c: (-(target * counts[c] % n), c))
    for c in order[:target - sum(q)]:
        q[c] += 1
    selected = []
    for c in (0, 1):
        if q[c] < 3 or q[c] > counts[c]:
            raise ValueError('Contribution resample cannot support three utility folds')
        idx = np.flatnonzero(pool.y == c)
        idx = idx[np.argsort(np.asarray(pool.row_ids)[idx], kind='stable')]
        selected.extend(np.random.default_rng(seed(outer_fold, 'contribution_class', b, c)).choice(
            idx, q[c], replace=False).tolist())
    return pool.take(sorted(selected))


def f1(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability vector')
    predicted = p >= .5
    tp = np.sum((y == 1) & predicted)
    denominator = 2 * tp + np.sum((y == 0) & predicted) + np.sum((y == 1) & ~predicted)
    return float(2 * tp / denominator) if denominator else 0.0


def stable_top(values, k=4):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 < k <= len(values):
        raise ValueError('Invalid ranking')
    path = sorted(range(len(values)), key=lambda j: (-values[j], j))
    return tuple(sorted(path[:k])), tuple(path)


def ass_score_path(scores, redundancy, eta=.1):
    """Bound formula: feature_path.forward_path, eta divided by original d."""
    scores, R = np.asarray(scores, float), np.asarray(redundancy, float)
    d = len(scores)
    if not d or not np.isfinite(eta) or eta < 0:
        raise ValueError('Nonnegative finite redundancy penalty required')
    if R.shape != (d, d) or not np.isfinite(R).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid redundancy dimensions or values')
    if not np.allclose(R, R.T, atol=1e-12, rtol=0) or np.any(np.diag(R) != 0) or np.any((R < 0) | (R > 1)):
        raise ValueError('Invalid redundancy matrix')
    path, remaining, increments = [], list(range(d)), []
    while remaining:
        delta = {j: float(scores[j] - eta / d * R[j, path].sum()) for j in remaining}
        chosen = min(remaining, key=lambda j: (-delta[j], j))
        path.append(chosen); remaining.remove(chosen); increments.append(delta[chosen])
    return tuple(path), increments


class FitOperations:
    """One reserve/finish pair around each explicit fit; no implicit retries.

    hook.reserve(kind, metadata) MUST commit before returning an operation ID.
    hook.finish(id, status, seconds, error=None) MUST commit completion.
    The hook and caller must both enable real fits. Test doubles advertise
    is_test_double=True and events explicitly carry simulated=True.
    """
    def __init__(self, backend, hook, outer_fold, phase='selection', enable_real_fits=False):
        if phase not in ('selection', 'final', 'audit', 'probe', 'replay'):
            raise ValueError('Invalid phase')
        self.backend, self.hook, self.outer_fold = backend, hook, outer_fold
        self.phase, self.enable_real_fits = phase, enable_real_fits
        self.prediction_seconds = 0.0

    def call(self, kind, method, role, coordinate, fn, n_rows, n_features):
        simulated = bool(getattr(self.backend, 'is_test_double', False))
        if not simulated and not (self.enable_real_fits and getattr(self.hook, 'enabled', False)):
            raise RuntimeError('Real fits disabled; A3.1 authorization, ledger and probe gates required')
        if kind not in ('classifier', 'encoder'):
            raise ValueError('Unbudgeted operation kind')
        meta = {'stage': 'A3', 'phase': self.phase, 'outer_fold': self.outer_fold,
                'method': method, 'role': role, 'coordinate': coordinate,
                'n_rows': n_rows, 'n_features': n_features, 'simulated': simulated}
        number = self.hook.reserve(kind, meta)
        start = time.perf_counter()
        try:
            result = fn()
        except BaseException as exc:
            self.hook.finish(number, 'failed', time.perf_counter() - start, repr(exc))
            raise
        self.hook.finish(number, 'succeeded', time.perf_counter() - start, None)
        return result

    def encoder(self, X, method, coordinate):
        return self.call('encoder', method, 'preprocessing', coordinate,
                         lambda: self.backend.fit_encoder(X), len(X), X.shape[1])

    def model(self, X, y, kind, model_seed, method, coordinate, C=1.0):
        return self.call('classifier', method, kind, coordinate,
                         lambda: self.backend.fit_model(X, y, kind, model_seed, C), len(X), X.shape[1])

    def fit_predict(self, training, validation_X, cols, model_seed, method, coordinate, kind='lr', C=1.0):
        cols = tuple(sorted(cols))
        enc = self.encoder(training.X[:, cols], method, coordinate)
        model = self.model(enc.transform(training.X[:, cols]), training.y, kind, model_seed, method, coordinate, C)
        start = time.perf_counter()
        p = model.predict_proba(enc.transform(validation_X[:, cols]))[:, list(model.classes_).index(1)]
        self.prediction_seconds += time.perf_counter() - start
        return p, model


class NumericEncoder:
    """HTRU2 finite-numeric restriction of historical model TabularEncoder."""
    def fit(self, X):
        X = np.asarray(X, float)
        if not len(X) or not np.isfinite(X).all():
            raise ValueError('Finite nonempty numeric matrix required')
        self.center_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.center_) / self.scale_


class SklearnBackend:
    is_test_double = False

    def fit_encoder(self, X):
        return NumericEncoder().fit(X)

    def fit_model(self, X, y, kind, model_seed, C=1.0):
        # Lazy imports and fit-only methods: never invoked by no-fit preflight.
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        if kind == 'rf':
            model = RandomForestClassifier(n_estimators=64, max_depth=4, min_samples_leaf=5,
                max_features=None, bootstrap=True, random_state=model_seed, n_jobs=1)
        elif kind in ('lr', 'l1'):
            model = LogisticRegression(C=C, penalty='l1' if kind == 'l1' else 'l2',
                solver='liblinear' if kind == 'l1' else 'lbfgs', max_iter=5000, tol=1e-6,
                fit_intercept=True, class_weight=None, random_state=model_seed)
        else:
            raise ValueError('Unsupported learner')
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(X, y)
        return model


def _rank_fit(pool, ops, kind, model_seed, method, coordinate, C=1.0):
    enc = ops.encoder(pool.X, method, coordinate)
    model = ops.model(enc.transform(pool.X), pool.y, kind, model_seed, method, coordinate, C)
    importance = np.abs(np.asarray(model.coef_)[0]) if kind == 'l1' else np.asarray(model.feature_importances_)
    chosen, path = stable_top(importance)
    return chosen, {'importance': importance.tolist(), 'path': list(path),
        'zero_count': int(np.sum(importance == 0)), 'tie_count': len(importance) - len(np.unique(importance))}


def l1_select(pool, ops, folds):
    details = []
    for c_index, C in enumerate(C_GRID):
        scores, rankings = [], []
        for j, (tr, va) in enumerate(folds):
            training = pool.take(tr, tuple(pool.row_ids[i] for i in va))
            chosen, diagnostic = _rank_fit(training, ops, 'l1', seed(ops.outer_fold, 'l1_inner', c_index, j),
                                          METHODS[2], ['C', c_index, 'inner', j], C)
            p, _ = ops.fit_predict(training, pool.X[va], chosen, seed(ops.outer_fold, 'inner_downstream', j),
                                   METHODS[2], ['C', c_index, 'inner', j, 'validation'])
            scores.append(f1(pool.y[va], p)); rankings.append(diagnostic)
        details.append({'C': C, 'fold_f1': scores, 'mean_f1': float(np.mean(scores)), 'rankings': rankings})
    winner = min(range(len(C_GRID)), key=lambda c: (-details[c]['mean_f1'], c))
    chosen, ranking = _rank_fit(pool, ops, 'l1', seed(ops.outer_fold, 'l1_rebuild', winner),
                               METHODS[2], ['rebuild', winner], C_GRID[winner])
    return chosen, {'chosen_C': C_GRID[winner], 'cv': details, 'rebuild': ranking}


def forward_select(pool, ops, folds):
    path, steps = [], []
    for step in range(4):
        candidates = []
        for col in range(pool.X.shape[1]):
            if col in path:
                continue
            scores = []
            for j, (tr, va) in enumerate(folds):
                training = pool.take(tr, tuple(pool.row_ids[i] for i in va))
                p, _ = ops.fit_predict(training, pool.X[va], (*path, col),
                    seed(ops.outer_fold, 'inner_downstream', j), METHODS[4], ['step', step, 'candidate', col, 'inner', j])
                scores.append(f1(pool.y[va], p))
            candidates.append({'feature': col, 'fold_f1': scores, 'mean_f1': float(np.mean(scores))})
        winner = min(candidates, key=lambda r: (-r['mean_f1'], r['feature']))
        path.append(winner['feature']); steps.append(candidates)
    return tuple(sorted(path)), {'path': path, 'steps': steps, 'negative_gain_stops': False}


class UtilityGame:
    def __init__(self, pool, ops, b):
        self.pool, self.ops, self.b = pool, ops, b
        self.folds = stratified_folds(pool, seed(ops.outer_fold, 'utility_split', b))
        self.cache, self.cache_hits, self.constant_evaluations = {}, 0, 0

    def __call__(self, subset):
        cols = tuple(sorted(subset))
        if len(set(cols)) != len(cols) or not set(cols) <= set(range(self.pool.X.shape[1])):
            raise ValueError('Invalid utility coalition')
        if cols in self.cache:
            self.cache_hits += 1
            return self.cache[cols]
        scores = []
        for j, (tr, va) in enumerate(self.folds):
            training = self.pool.take(tr, tuple(self.pool.row_ids[i] for i in va))
            if cols:
                p, _ = self.ops.fit_predict(training, self.pool.X[va], cols,
                    seed(self.ops.outer_fold, 'utility_model', self.b, j), METHODS[0],
                    ['game', self.b, 'coalition', sum(1 << c for c in cols), 'utility_fold', j])
            else:
                p = np.full(len(va), float(training.y.mean()))
                self.constant_evaluations += 1
            scores.append(f1(self.pool.y[va], p))
        self.cache[cols] = float(np.mean(scores))
        return self.cache[cols]


def redundancy(pool, ops):
    # scipy rankdata uses average ranks, matching historical pandas.rank average.
    from scipy.stats import rankdata
    enc = ops.encoder(pool.X, METHODS[0], ['redundancy'])
    ranks = rankdata(enc.transform(pool.X), method='average', axis=0)
    ranks -= ranks.mean(axis=0)
    norm = np.sqrt((ranks ** 2).sum(axis=0)); denominator = np.outer(norm, norm)
    R = np.divide(ranks.T @ ranks, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    R = np.clip(np.abs(R), 0, 1); np.fill_diagonal(R, 0)
    return R


def select_all(pool, ops):
    """All seven arms selected from one outer training pool; no test inputs."""
    if len(pool.features) != 8:
        raise ValueError('Frozen A3 requires eight HTRU2 original variables')
    started = time.perf_counter()
    games, estimates, variances = [], [], []
    for b in range(4):
        sample = contribution_pool(pool, ops.outer_fold, b)
        game = UtilityGame(sample, ops, b)
        approximate = permutation_shapley(game, tuple(range(8)), 256, seed(ops.outer_fold, 'permutation', b))
        estimates.append(approximate['phi']); variances.append(approximate['mc_variance_of_mean'])
        games.append({'b': b, 'training_ids': sample.row_ids, 'coalitions': len(game.cache),
            'cache_hits': game.cache_hits, 'constant_evaluations': game.constant_evaluations,
            'phi': approximate['phi'].tolist(), 'mc_variance': approximate['mc_variance_of_mean'].tolist()})
    stats = corrected_scores(estimates, variances, .5)
    R = redundancy(pool, ops)
    path, increments = ass_score_path(stats['score'], R, .1)
    mean_chosen, mean_path = stable_top(stats['mean'])
    chosen = {METHODS[0]: tuple(sorted(path[:4])), METHODS[1]: mean_chosen}
    diagnostic = {METHODS[0]: {'path': list(path), 'increments': increments, 'games': games,
        'statistics': {key: value.tolist() for key, value in stats.items()}, 'redundancy': R.tolist()},
        METHODS[1]: {'path': list(mean_path), 'alias': '000', 'independent_method_count': 1}}
    folds = stratified_folds(pool, seed(ops.outer_fold, 'inner_split'))
    chosen[METHODS[2]], diagnostic[METHODS[2]] = l1_select(pool, ops, folds)
    chosen[METHODS[3]], diagnostic[METHODS[3]] = _rank_fit(pool, ops, 'rf', seed(ops.outer_fold, 'rf_model'), METHODS[3], ['ranking'])
    chosen[METHODS[4]], diagnostic[METHODS[4]] = forward_select(pool, ops, folds)
    chosen[METHODS[5]] = tuple(sorted(np.random.default_rng(seed(ops.outer_fold, 'random_features')).choice(8, 4, replace=False)))
    chosen[METHODS[6]] = tuple(range(8))
    return {'selected': chosen, 'diagnostics': diagnostic, 'training_ids': pool.row_ids,
        'outer_fold': ops.outer_fold, 'rho': 1.0, 'seconds': time.perf_counter() - started,
        'prediction_seconds': ops.prediction_seconds, 'fitted_on_outer_test': False}


def final_evaluate(pool, heldout_X, heldout_y, heldout_ids, selections, ops):
    """Must be invoked only after selections are saved/frozen by a future runner."""
    heldout_X, heldout_y = np.asarray(heldout_X, float), np.asarray(heldout_y)
    if ops.phase not in ('final', 'audit', 'replay'):
        raise ValueError('Final evaluation requires explicit final/audit/replay phase')
    if (set(pool.row_ids) & set(heldout_ids) or len(heldout_ids) != len(heldout_X)
            or len(set(heldout_ids)) != len(heldout_ids)):
        raise ValueError('Final train/test identity leakage')
    if (heldout_X.ndim != 2 or heldout_X.shape[1] != 8 or heldout_y.shape != (len(heldout_X),)
            or not len(heldout_X) or not np.isfinite(heldout_X).all() or not set(heldout_y) <= {0, 1}):
        raise ValueError('Invalid held-out data dimensions')
    if set(selections) != set(METHODS):
        raise ValueError('Every frozen comparison arm required')
    from sklearn.metrics import roc_auc_score, average_precision_score
    results = {}
    for method in METHODS:
        cols = tuple(selections[method])
        expected_k = 8 if method == METHODS[6] else 4
        if len(cols) != expected_k or len(set(cols)) != len(cols) or not set(cols) <= set(range(8)):
            raise ValueError('Invalid fixed-k selection')
        started = time.perf_counter(); prediction_start = ops.prediction_seconds
        p, _ = ops.fit_predict(pool, heldout_X, cols, seed(ops.outer_fold, 'final_model'), method, ['final'])
        results[method] = {'features': [pool.features[i] for i in sorted(cols)], 'actual_k': len(cols),
            'f1': f1(heldout_y, p), 'auc': float(roc_auc_score(heldout_y, p)) if len(set(heldout_y)) == 2 else None,
            'ap': float(average_precision_score(heldout_y, p)) if np.any(heldout_y == 1) else None,
            'probabilities': p.tolist(), 'seconds': time.perf_counter() - started,
            'prediction_seconds': ops.prediction_seconds - prediction_start}
    return results
