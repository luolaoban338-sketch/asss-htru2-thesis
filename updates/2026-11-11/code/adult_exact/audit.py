"""Independent categorical/geometry reconstruction and bounded direct refits.

This module does not call the main selector, main encoder, main sampler,
eight-arm decision code, or main utility evaluator to establish correctness.
It shares only guarded operation accounting, immutable I/O and data loading.
"""
from pathlib import Path
from collections import Counter
import math
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from private_science.engine import (load_data, positions, seed_for, fingerprint, digest, cached, commit,
                                   checked_fit, checked_predict, assert_pair)
from adapter import ARMS, METHODS, validate_config, verify_decision_lock


def equal(actual, expected, label, atol=1e-11):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected): raise AssertionError(label + ': dictionary keys differ')
        for key in actual: equal(actual[key], expected[key], label + '/' + str(key), atol)
    elif isinstance(actual, (list, tuple, np.ndarray)) or isinstance(expected, (list, tuple, np.ndarray)):
        if len(actual) != len(expected): raise AssertionError(label + ': length differs')
        for i, (a, b) in enumerate(zip(actual, expected)): equal(a, b, label + '/' + str(i), atol)
    elif isinstance(actual, (int, float, np.number)) and not isinstance(actual, bool):
        if not np.isfinite(actual) or not np.isfinite(expected) or not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=0):
            raise AssertionError(label + ': numeric mismatch')
    elif actual != expected: raise AssertionError(label + ': value differs')


class AuditEncoder:
    """Independent original-variable encoder, including categorical blocks."""
    def __init__(self, numeric, categorical, session, geometry=False):
        self.numeric, self.categorical, self.session, self.geometry = tuple(numeric), tuple(categorical), session, geometry

    def fit(self, frame):
        self.session.require_execution()
        self.order = tuple(frame.columns); self.parameters = {}; self.blocks = {}; cursor = 0
        if set(self.order) != set(self.numeric) | set(self.categorical): raise AssertionError('Audit schema mismatch')
        for name in self.order:
            if name in self.numeric:
                values = frame[name].to_numpy(dtype=float)
                observed = values[~np.isnan(values)]
                impute = float(np.median(observed)) if len(observed) else 0.
                values = np.where(np.isnan(values), impute, values)
                if self.geometry:
                    center = float(np.median(values)); qs = np.quantile(values, [.25, .75, .01, .99])
                    scale, low, high = float(qs[1] - qs[0]), float(qs[2]), float(qs[3])
                else: center, scale, low, high = float(np.mean(values)), float(np.std(values)), None, None
                self.parameters[name] = {'kind': 'numeric', 'impute': impute, 'center': center,
                    'scale': 1. if scale == 0 else scale, 'zero_scale': bool(scale == 0),
                    'clip_low': low, 'clip_high': high, 'all_missing': len(observed) == 0}
                width = 1
            else:
                original = frame[name].tolist(); observed = [str(x) for x in original if not pd.isna(x)]
                counts = Counter(observed)
                impute = min(counts, key=lambda value: (-counts[value], value)) if counts else '__MISSING__'
                categories = sorted(set(impute if pd.isna(x) else str(x) for x in original))
                width = len(categories)
                self.parameters[name] = {'kind': 'categorical', 'impute': impute, 'categories': categories,
                    'block_scale': math.sqrt(width) if self.geometry else 1., 'all_missing': not observed}
            self.blocks[name] = list(range(cursor, cursor + width)); cursor += width
        self.width = cursor; self.n_rows = len(frame)
        return self

    def transform(self, frame):
        self.session.require_execution()
        if tuple(frame.columns) != self.order: raise AssertionError('Audit feature order mismatch')
        out = np.zeros((len(frame), self.width), float)
        for name in self.order:
            p = self.parameters[name]; columns = self.blocks[name]
            if p['kind'] == 'numeric':
                values = frame[name].to_numpy(dtype=float)
                values = np.where(np.isnan(values), p['impute'], values)
                if self.geometry: values = np.minimum(np.maximum(values, p['clip_low']), p['clip_high'])
                out[:, columns[0]] = (values - p['center']) / p['scale']
            else:
                values = [p['impute'] if pd.isna(x) else str(x) for x in frame[name].tolist()]
                category_index = {value: i for i, value in enumerate(p['categories'])}
                for row, value in enumerate(values):
                    if value in category_index: out[row, columns[category_index[value]]] = 1. / p['block_scale']
        if not np.isfinite(out).all(): raise AssertionError('Nonfinite independent encoded data')
        return out


def _encoder(data, features, session, geometry=False):
    return AuditEncoder([f for f in features if f in data.numeric], [f for f in features if f in data.categorical], session, geometry)


def independent_sample(data, raw, method, seed, config, session, metadata):
    session.require_execution(); raw = positions(raw)
    if method == 'full': return raw.copy()
    if method != 'robust_half': raise AssertionError('Unexpected sampler')
    labels = data.y[raw]; target = math.ceil(len(raw) / 2)
    sizes = {c: int(np.sum(labels == c)) for c in (0, 1)}
    quota = {c: target * sizes[c] // len(raw) for c in (0, 1)}
    rank = sorted((0, 1), key=lambda c: (-(target * sizes[c] % len(raw)), c))
    for c in rank[:target - sum(quota.values())]: quota[c] += 1
    if min(quota.values()) < 2: raise AssertionError('Infeasible class quota')
    enc = _encoder(data, tuple(data.X.columns), session, True)
    checked_fit(session, 'encoder', enc, data.X.iloc[raw], metadata=metadata)
    Z = enc.transform(data.X.iloc[raw]); chosen = []
    for c in (0, 1):
        members = sorted(np.flatnonzero(labels == c), key=lambda i: data.row_ids[raw[i]])
        members = np.asarray(members, int)
        clusters = min(config['sampler']['max_clusters'], math.isqrt(len(members)), quota[c], len(np.unique(Z[members], axis=0)))
        estimator = KMeans(n_clusters=clusters, init='k-means++', n_init=config['sampler']['n_init'],
            max_iter=config['sampler']['max_iter'], algorithm=config['sampler']['algorithm'], random_state=seed_for(seed, 'kmeans', c))
        checked_fit(session, 'kmeans', estimator, Z[members], metadata={**metadata, 'class': c})
        groups = {int(label): np.flatnonzero(estimator.labels_ == label) for label in set(estimator.labels_)}
        ordered = sorted(groups, key=lambda label: min(data.row_ids[raw[members[groups[label]]]]))
        capacities = [len(groups[label]) for label in ordered]; allocation = [1] * len(ordered)
        while sum(allocation) < quota[c]:
            active = [j for j in range(len(ordered)) if allocation[j] < capacities[j]]
            remaining = quota[c] - sum(allocation); total_weight = sum(capacities[j] for j in active)
            for j in active: allocation[j] += min(remaining * capacities[j] // total_weight, capacities[j] - allocation[j])
            left = quota[c] - sum(allocation)
            eligible = [j for j in active if allocation[j] < capacities[j]]
            order = sorted(eligible, key=lambda j: (-(remaining * capacities[j] % total_weight), j))
            for j in order[:left]: allocation[j] += 1
        for label, amount in zip(ordered, allocation):
            group = groups[label]
            ordered_members = sorted(group, key=lambda i: (float(np.linalg.norm(Z[members[i]] - estimator.cluster_centers_[label])), data.row_ids[raw[members[i]]]))
            ranks = [(len(group) - 1) // 2] if amount == 1 else [j * (len(group) - 1) // (amount - 1) for j in range(amount)]
            chosen.extend(int(raw[members[ordered_members[j]]]) for j in ranks)
    return np.asarray(sorted(chosen), int)


def independent_redundancy(data, rows, session, metadata):
    enc = _encoder(data, tuple(data.X.columns), session)
    checked_fit(session, 'encoder', enc, data.X.iloc[rows], metadata=metadata)
    Z = enc.transform(data.X.iloc[rows]); columns = []
    for i in range(Z.shape[1]):
        values = rankdata(Z[:, i], method='average'); columns.append(values - values.mean())
    R = np.zeros((12, 12)); names = tuple(data.X.columns)
    for i in range(12):
        for j in range(i + 1, 12):
            values = []
            for a in enc.blocks[names[i]]:
                for b in enc.blocks[names[j]]:
                    divisor = math.sqrt(float(np.dot(columns[a], columns[a]) * np.dot(columns[b], columns[b])))
                    values.append(abs(float(np.dot(columns[a], columns[b]))) / divisor if divisor else 0.)
            R[i, j] = R[j, i] = min(1., max(values))
    return R


def independent_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, float); predicted = p >= .5
    tp = int(np.sum((y == 1) & predicted)); fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted)); tn = int(np.sum((y == 0) & ~predicted))
    denom = 2 * tp + fp + fn; positives = tp + fn; negatives = tn + fp
    auc = ((rankdata(p)[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)) if positives and negatives else None
    return {'f1': 2 * tp / denom if denom else 0., 'roc_auc': float(auc) if auc is not None else None,
            'accuracy': (tp + tn) / len(y), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'n': len(y)}


def direct_cycle(data, rows, evaluation, features, seed, config, session, metadata):
    session.require_execution(); rows, evaluation = positions(rows), positions(evaluation)
    assert_pair(data, rows, evaluation); features = tuple(f for f in data.X.columns if f in features)
    begin = time.perf_counter()
    metadata = session.bind_classifier(rows, evaluation, features, metadata)
    enc = _encoder(data, features, session)
    checked_fit(session, 'encoder', enc, data.X.iloc[rows][list(features)], metadata=metadata)
    learner = LogisticRegression(**config['learner_parameters'], random_state=int(seed))
    checked_fit(session, 'classifier', learner, enc.transform(data.X.iloc[rows][list(features)]), data.y[rows], metadata=metadata)
    p = checked_predict(session, learner, enc.transform(data.X.iloc[evaluation][list(features)]))[:, list(learner.classes_).index(1)]
    metrics = independent_metrics(data.y[evaluation], p)
    elapsed = time.perf_counter() - begin
    session.record_event('audit_complete_cycle', metadata, elapsed)
    return p, metrics, elapsed


def audit_game_arithmetic(game, frozen, method):
    permutations = np.asarray(game['permutation']['permutations'], int)
    equal(permutations, frozen['permutations'][method], 'frozen permutations', 0)
    coalition = {r['mask']: r for r in game['coalitions']}
    if len(coalition) != len(game['coalitions']): raise AssertionError('Duplicate coalition')
    marginals = np.zeros((len(permutations), 12))
    for m, order in enumerate(permutations):
        mask = 0; prior = coalition[0]['utility']
        for feature in order:
            mask |= 1 << int(feature); value = coalition[mask]['utility']
            marginals[m, feature] = value - prior; prior = value
    for r in coalition.values(): equal(r['utility'], float(np.mean(r['fold_f1'])), 'utility fold mean')
    equal(game['permutation']['marginals'], marginals, 'marginals')
    equal(game['permutation']['phi'], marginals.mean(axis=0), 'phi')
    equal(game['permutation']['mc_variance_of_mean'], marginals.var(axis=0, ddof=1) / len(permutations), 'mc variance')
    expected_masks = set(frozen['nonempty_coalition_masks'][method])
    if set(coalition) != expected_masks | {0}: raise AssertionError('Coalition cache coverage differs from frozen permutations')
    return marginals.mean(axis=0), marginals.var(axis=0, ddof=1) / len(permutations)


def _audit_selector(data, context, method, config, source, destination, session):
    source_file = source / 'selector.json'; saved = cached(source_file)
    if saved is None: raise AssertionError('Missing selector')
    bound = fingerprint({'source_selector': digest(source_file), 'games': saved['game_hashes'], 'context': context})
    previous = cached(destination)
    if previous is not None:
        if previous['input_fingerprint'] != bound: raise AssertionError('Audit source changed')
        for rel, sha in saved['game_hashes'].items():
            if digest(source / rel) != sha: raise AssertionError('Audited game changed')
        return previous
    meta = {'fold': context['fold'], 'phase': 'audit_selector_setup', 'method': method, 'context': context['partition_id']}
    rows = positions(context['train'])
    setup_started = time.perf_counter()
    selected = independent_sample(data, rows, method, context['selector_sampler_seed'], config, session, meta)
    equal(saved['selected_positions'], selected, 'independent selector sampler', 0)
    equal(saved['actual_rho'], len(selected) / len(rows), 'selector rho')
    R = independent_redundancy(data, selected, session, meta)
    equal(saved['redundancy'], R, 'categorical redundancy')
    session.record_event('resource_rate', {'rate': 't_selector_setup'}, time.perf_counter() - setup_started)
    estimates, mc, direct_results = [], [], []
    for frozen in context['games']:
        session.checkpoint(); game = cached(source / f'game-{frozen["resample_index"]}.json')
        if game is None: raise AssertionError('Missing game artifact')
        estimate, variance = audit_game_arithmetic(game, frozen, method); estimates.append(estimate); mc.append(variance)
        masks = sorted(r['mask'] for r in game['coalitions'] if 0 < r['mask'] < 4095)
        if not masks: raise AssertionError('No proper nonempty audit coalition')
        chosen_mask = masks[0]; features = tuple(f for i, f in enumerate(data.X.columns) if chosen_mask & (1 << i))
        expected = next(r for r in game['coalitions'] if r['mask'] == chosen_mask)
        game_setup_seconds = 0.
        for u, fold in enumerate(frozen['utility_folds']):
            raw, valid = positions(fold['train_positions']), positions(fold['validation_positions'])
            assert_pair(data, raw, valid)
            game_setup_started = time.perf_counter()
            selected_utility = independent_sample(data, raw, method, fold['sampler_seed'], config, session,
                {**meta, 'phase': 'audit_game_setup', 'resample': frozen['resample_index'], 'utility_fold': u})
            game_setup_seconds += time.perf_counter() - game_setup_started
            equal(game['folds'][u]['selected_positions'], selected_utility, 'utility selected identities', 0)
            _, metrics, elapsed = direct_cycle(data, selected_utility, valid, features, fold['model_seed'], config,
                session, {**meta, 'phase': 'audit_utility', 'mask': chosen_mask, 'resample': frozen['resample_index'], 'utility_fold': u})
            equal(metrics['f1'], expected['fold_f1'][u], 'independent utility refit F1')
            direct_results.append({'resample': frozen['resample_index'], 'utility_fold': u, 'mask': chosen_mask,
                                   'f1': metrics['f1'], 'cycle_seconds': elapsed})
        session.record_event('resource_rate', {'rate': 't_game_setup'}, game_setup_seconds)
    estimates, mc = np.asarray(estimates), np.asarray(mc)
    mean = estimates.mean(axis=0); observed = estimates.var(axis=0, ddof=1); raw = observed - mc.mean(axis=0)
    score = mean - config['lambda_grid'][0] * np.sqrt(np.maximum(raw, 0.))
    equal(saved['scores']['mean'], mean, 'mean contribution'); equal(saved['scores']['observed_variance'], observed, 'observed variance')
    equal(saved['scores']['mean_mc_variance'], mc.mean(axis=0), 'mean MC variance')
    equal(saved['scores']['raw_corrected_variance'], raw, 'raw corrected variance')
    equal(saved['scores']['score'], score, 'corrected score')
    chosen, remaining, expected_path = [], list(range(12)), []
    for k in range(12):
        ranked = [(float(score[j] - config['eta_grid'][0] * sum(R[j, x] for x in chosen) / 12), j) for j in remaining]
        j = sorted(ranked, key=lambda pair: (-pair[0], pair[1]))[0][1]; chosen.append(j); remaining.remove(j)
        expected_path.append([f for i, f in enumerate(data.X.columns) if i in chosen])
    equal(saved['paths'], expected_path, 'independent greedy path', 0)
    return commit(destination, {'status': 'passed', 'input_fingerprint': bound, 'context': context['partition_id'],
                                'method': method, 'direct_classifier_refits': 12, 'direct_results': direct_results,
                                'categorical_geometry_independently_reconstructed': True})


def independent_decisions(tables, config):
    rows = []
    for i in range(24):
        matches = [[r for r in table if r['candidate_id'] == i] for table in tables]
        if any(len(m) != 1 for m in matches): raise AssertionError('Candidate coverage incomplete')
        records = [m[0] for m in matches]; method = METHODS[i // 12]; k = i % 12 + 1
        if any(r['method'] != method or r['k'] != k for r in records): raise AssertionError('Candidate identity mismatch')
        values = [r['f1'] for r in records]; rho = sum(r['actual_rho'] for r in records) / 3
        rows.append({'candidate_id': i, 'method': method, 'k': k, 'fold_f1': values,
                     'mean_f1': sum(values) / 3, 'mean_actual_rho': rho, 'cell_fraction': rho * k / 12})
    threshold = rows[11]['mean_f1'] - config['epsilon']; tol = config['numeric_tolerance']
    def select(pool):
        eligible = [r for r in pool if r['mean_f1'] + tol >= threshold]
        cost = min(r['cell_fraction'] for r in eligible); eligible = [r for r in eligible if r['cell_fraction'] - cost <= tol]
        size = min(r['k'] for r in eligible); eligible = [r for r in eligible if r['k'] == size]
        score = max(r['mean_f1'] for r in eligible); eligible = [r for r in eligible if score - r['mean_f1'] <= tol]
        return sorted(eligible, key=lambda r: r['candidate_id'])[0]
    return {'baseline_inner_f1': rows[11]['mean_f1'], 'threshold': threshold,
        'selections': {'full': rows[11], 'sample_only': select([r for r in rows if r['k'] == 12]),
                       'feature_only': select(rows[:12]), 'joint': select(rows)},
        'candidate_scores': rows, 'reference_used_in_decisions': False}


def _check_model(data, model, selected, evaluation, features, seed):
    equal(model['selected_positions'], positions(selected), 'model training IDs', 0)
    equal(model['evaluation_positions'], positions(evaluation), 'model evaluation IDs', 0)
    equal(model['features'], features, 'model features', 0); equal(model['seed'], seed, 'model seed', 0)
    assert_pair(data, selected, evaluation)
    p = np.asarray(model['probabilities'], float)
    if p.shape != (len(evaluation),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise AssertionError('Invalid stored probabilities')
    metrics = independent_metrics(data.y[positions(evaluation)], p)
    for key, value in metrics.items(): equal(model['metrics'][key], value, 'stored metric/' + key)
    blocks = model['encoder']['blocks']; parameters = model['encoder']['parameters']; cursor = 0
    for feature in features:
        width = 1 if feature in data.numeric else len(parameters[feature]['categories'])
        equal(blocks[feature], list(range(cursor, cursor + width)), 'complete original-variable block', 0); cursor += width
    equal(model['encoder']['encoded_d'], cursor, 'encoded dimension', 0)


def audit_fold(config, paths, partitions, fold, role, session):
    session.require_execution(); validate_config(config)
    if session.role != 'audit' or role not in ('main', 'replay') or session.task != f'{role}-fold-{fold}':
        raise PermissionError('Audit role/task mismatch')
    if role == 'replay' and fold != 0: raise ValueError('Only fold-0 replay can be audited')
    source = Path(paths['run_root']) / role / f'fold-{fold}'
    dest = Path(session.artifact_path('')); bundle = cached(source / 'bundle.json')
    if bundle is None: raise AssertionError('Complete source bundle required before audit')
    for rel, sha in bundle['artifact_sha256'].items():
        if digest(source / rel) != sha: raise AssertionError('Source bundle hash mismatch')
    verify_decision_lock(source)
    data = load_data(config, paths); outer = next(o for o in partitions['outer_folds'] if o['fold'] == fold)
    test = positions(outer['test']); contexts = outer['inner_context_ids'] + [outer['refit_context_id']]
    selector_audits = []
    for c, key in enumerate(contexts):
        label = f'inner-{c}' if c < 3 else 'refit'; context = partitions['contexts'][key]
        for method in METHODS:
            selector_audits.append(_audit_selector(data, context, method, config, source / label / method,
                                                   dest / label / f'{method}_audit.json', session))
    tables = []
    for i, key in enumerate(outer['inner_context_ids']):
        context = partitions['contexts'][key]; table = cached(source / f'inner-{i}' / 'candidates.json')
        for record in table:
            model_path = source / record['model_path']
            if digest(model_path) != record['model_sha256']: raise AssertionError('Candidate model changed')
            model = cached(model_path); selector = cached(source / f'inner-{i}' / record['method'] / 'selector.json')
            _check_model(data, model, selector['selected_positions'], context['validation']['positions'],
                         selector['paths'][record['k'] - 1], outer['inner_model_seeds'][i])
            equal(record['f1'], model['metrics']['f1'], 'candidate F1')
        tables.append(table)
    decisions = independent_decisions(tables, config)
    equal(cached(source / 'decisions.json'), decisions, 'independent four-arm decisions')
    release = cached(source / 'outer_release.json')
    if tuple(release['arms']) != ARMS: raise AssertionError('Outer arm scope changed')
    final_audits = []
    for arm in (*ARMS, 'complete_outer_training_reference'):
        if arm in ARMS:
            r = release['arms'][arm]; candidate = decisions['selections'][arm]
            selector = cached(source / 'refit' / candidate['method'] / 'selector.json')
            selected = positions(selector['selected_positions']); features = selector['paths'][candidate['k'] - 1]
            model_path = source / r['model_path']; seed = outer['outer_model_seed']; phase = 'audit_final'
            equal(r['candidate'], candidate, 'published candidate')
            equal(r['features'], features, 'published feature set', 0)
            equal(r['selected_rows'], len(selected), 'published selected rows', 0)
            equal(r['pool_rows'], len(outer['pool']['positions']), 'published pool rows', 0)
            equal(r['original_outer_train_rows'], len(outer['original_train']['positions']), 'published original rows', 0)
            equal(r['model_sha256'], digest(model_path), 'published model hash', 0)
            equal(r['rho_pool'], len(selected) / len(outer['pool']['positions']), 'pool denominator')
            equal(r['rho_original_outer_train'], len(selected) / len(outer['original_train']['positions']), 'original denominator')
            equal(r['cell_fraction_pool'], r['rho_pool'] * len(features) / 12, 'pool cells')
            equal(r['cell_fraction_original_outer_train'], r['rho_original_outer_train'] * len(features) / 12, 'original cells')
        else:
            selected = positions(outer['original_train']); features = list(data.X.columns); seed = outer['reference_model_seed']
            model_path = source / release[arm]['model_path']; phase = 'audit_reference'
        model = cached(model_path); _check_model(data, model, selected, test, features, seed)
        published_metrics = release['arms'][arm]['metrics'] if arm in ARMS else release[arm]['metrics']
        equal(published_metrics, model['metrics'], 'published model metrics')
        audit_path = dest / f'{arm}_direct.json'; prior = cached(audit_path)
        if prior is None:
            p, metrics, elapsed = direct_cycle(data, selected, test, features, seed, config, session,
                                               {'phase': phase, 'fold': fold, 'arm': arm})
            equal(p, model['probabilities'], 'independent outer probabilities')
            for key, value in metrics.items(): equal(model['metrics'][key], value, 'independent outer metric/' + key)
            prior = commit(audit_path, {'status': 'passed', 'source_sha256': digest(model_path),
                         'classifier_refits': 1, 'cycle_seconds': elapsed,
                         'max_probability_error': float(np.max(np.abs(p - np.asarray(model['probabilities']))))})
        elif prior['source_sha256'] != digest(model_path): raise AssertionError('Direct audit source changed')
        final_audits.append(prior)
    reference_record = release['complete_outer_training_reference']
    if reference_record['used_in_decisions'] is not False: raise AssertionError('Reference entered decision policy')
    equal(reference_record['rows'], len(outer['original_train']['positions']), 'reference training count', 0)
    equal(reference_record['model_sha256'], digest(source / reference_record['model_path']), 'reference model hash', 0)
    pool_f1 = release['arms']['full']['metrics']['f1']; reference_f1 = reference_record['metrics']['f1']
    equal(release['pool_full_loss_vs_complete_outer_training'], reference_f1 - pool_f1, 'pool-full reference loss')
    equal(release['outer_test_positions'], test, 'released complete outer test identities', 0)
    if release['official_adult_test_used'] is not False: raise AssertionError('Official test used')
    for record in release['arms'].values():
        loss = pool_f1 - record['metrics']['f1']
        equal(record['loss_vs_pool_full'], loss, 'published pool-full loss')
        equal(record['loss_vs_complete_outer_training_reference'], reference_f1 - record['metrics']['f1'], 'published reference loss')
        equal(record['pool_tolerance_violation'], loss > config['epsilon'] + config['numeric_tolerance'], 'published violation', 0)
    replay = None
    if role == 'replay': replay = compare_replay(config, paths, fold)
    return commit(dest / 'audit.json', {'status': 'passed', 'source_role': role, 'fold': fold,
        'source_bundle_sha256': digest(source / 'bundle.json'), 'direct_classifier_refits': 101,
        'selector_audits': selector_audits, 'final_audits': final_audits,
        'categorical_geometry_and_game_memberships_reconstructed': True,
        'cold_replay_comparison': replay, 'automatic_resume': False})


def compare_replay(config, paths, fold=0):
    """Read-only substantive equality; timing and operation IDs are excluded."""
    left = Path(paths['run_root']) / 'main' / f'fold-{fold}'; right = Path(paths['run_root']) / 'replay' / f'fold-{fold}'
    if cached(left / 'bundle.json') is None or cached(right / 'bundle.json') is None:
        raise AssertionError('Both complete bundles required for cold replay comparison')
    equal(cached(left / 'decisions.json'), cached(right / 'decisions.json'), 'replay decisions')
    checked = 0
    for path in left.rglob('selector.json'):
        other = right / path.relative_to(left); a, b = cached(path), cached(other)
        for key in ('selected_positions', 'actual_rho', 'redundancy', 'scores', 'paths'):
            equal(a[key], b[key], 'replay selector/' + key)
        for game in path.parent.glob('game-*.json'):
            a_game, b_game = cached(game), cached(right / game.relative_to(left))
            for key in ('record', 'method', 'permutation', 'folds', 'coalitions'):
                equal(a_game[key], b_game[key], 'replay game/' + key)
            checked += 1
    for path in left.rglob('mask-*.json'):
        a, b = cached(path), cached(right / path.relative_to(left))
        for key in ('selected_positions', 'evaluation_positions', 'features', 'seed', 'encoder', 'probabilities', 'metrics'):
            equal(a[key], b[key], 'replay model/' + key)
        checked += 1
    a = cached(left / 'outer_models' / 'complete_outer_training_reference.json')
    b = cached(right / 'outer_models' / 'complete_outer_training_reference.json')
    for key in ('selected_positions', 'evaluation_positions', 'features', 'seed', 'encoder', 'probabilities', 'metrics'):
        equal(a[key], b[key], 'replay reference/' + key)
    a_release, b_release = cached(left / 'outer_release.json'), cached(right / 'outer_release.json')
    for arm in ARMS:
        keys = set(a_release['arms'][arm]) - {'model_sha256'}
        equal({k: a_release['arms'][arm][k] for k in keys}, {k: b_release['arms'][arm][k] for k in keys}, 'replay released arm/' + arm)
    equal(a_release['pool_full_loss_vs_complete_outer_training'], b_release['pool_full_loss_vs_complete_outer_training'], 'replay release pool loss')
    return {'status': 'passed', 'substantive_artifacts_checked': checked + 2, 'timing_excluded': True}
