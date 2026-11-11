"""Future four-arm Adult runner. Importing this module performs no experiments.

The old eight-arm decision entry point is neither imported nor called. Every
execution requires a live runtime session, even when all artifacts are cached.
"""
from pathlib import Path
import numpy as np
from private_science.engine import (load_data, positions, fingerprint, digest, read, commit, cached,
                                   build_selector, model_cycle, assert_pair, serial)

ARMS = ('full', 'sample_only', 'feature_only', 'joint')
METHODS = ('full', 'robust_half')


def validate_config(config):
    expected = {'methods': list(METHODS), 'arms': list(ARMS), 'B': 4, 'M': 64,
                'inner_folds': 3, 'utility_folds': 3, 'lambda_grid': [0.5], 'eta_grid': [0.1],
                'subsample_rho': 0.8, 'epsilon': 0.01, 'numeric_tolerance': 1e-12}
    for key, wanted in expected.items():
        if config.get(key) != wanted: raise ValueError('Unfrozen reduced design: ' + key)
    if config.get('official_adult_test_used') is not False: raise ValueError('Official Adult test is excluded')
    return True


def choose(rows, threshold, tolerance):
    eligible = [r for r in rows if r['mean_f1'] >= threshold - tolerance]
    if not eligible: raise ValueError('No feasible candidate; tolerance cannot be relaxed')
    minimum = min(r['cell_fraction'] for r in eligible)
    eligible = [r for r in eligible if r['cell_fraction'] <= minimum + tolerance]
    k = min(r['k'] for r in eligible); eligible = [r for r in eligible if r['k'] == k]
    best = max(r['mean_f1'] for r in eligible)
    return min((r for r in eligible if r['mean_f1'] >= best - tolerance), key=lambda r: r['candidate_id'])


def make_four_decisions(evaluations, config, d=12):
    """Pure arithmetic, exactly four decisions; no restricted/sequential extras."""
    validate_config(config)
    if d != 12: raise ValueError('Only twelve original variables are frozen')
    if len(evaluations) != 3: raise ValueError('Exactly three inner evaluations required')
    if any(len(rows) != 24 or {r['candidate_id'] for r in rows} != set(range(24)) for rows in evaluations):
        raise ValueError('Each inner fold requires exactly 24 unique candidates')
    aggregated = []
    for candidate_id, (method, k) in enumerate((m, k) for m in METHODS for k in range(1, d + 1)):
        found = []
        for rows in evaluations:
            matches = [r for r in rows if r['candidate_id'] == candidate_id]
            if len(matches) != 1 or matches[0]['method'] != method or matches[0]['k'] != k:
                raise ValueError('Candidate table missing, duplicated, or reordered')
            found.append(matches[0])
        values = np.asarray([r['f1'] for r in found], float); rho = np.asarray([r['actual_rho'] for r in found], float)
        if not np.isfinite(values).all() or not np.isfinite(rho).all() or np.any((values < 0) | (values > 1)) or np.any((rho <= 0) | (rho > 1)):
            raise ValueError('Candidate metrics or sampling rate invalid')
        aggregated.append({'candidate_id': candidate_id, 'method': method, 'k': k,
                           'fold_f1': values.tolist(), 'mean_f1': float(values.mean()),
                           'mean_actual_rho': float(rho.mean()), 'cell_fraction': float(rho.mean() * k / d)})
    full = next(r for r in aggregated if r['method'] == 'full' and r['k'] == d)
    threshold = full['mean_f1'] - config['epsilon']; tol = config['numeric_tolerance']
    selections = {
        'full': full,
        'sample_only': choose([r for r in aggregated if r['k'] == d], threshold, tol),
        'feature_only': choose([r for r in aggregated if r['method'] == 'full'], threshold, tol),
        'joint': choose(aggregated, threshold, tol),
    }
    if tuple(selections) != ARMS: raise AssertionError('Only the four frozen arms are allowed')
    return {'baseline_inner_f1': full['mean_f1'], 'threshold': threshold, 'selections': selections,
            'candidate_scores': aggregated, 'reference_used_in_decisions': False}


def _root(session, output_root):
    expected = Path(session.artifact_path('')).resolve()
    if output_root is not None and Path(output_root).resolve() != expected:
        raise PermissionError('Output root must equal the runtime-owned role/task directory')
    return expected


def _model(data, selected, evaluation, features, seed, config, session, dest, metadata):
    bound = fingerprint({'data': data.source_hash, 'selected': positions(selected), 'evaluation': positions(evaluation),
                         'features': features, 'seed': seed, 'config': config})
    prior = cached(dest)
    if prior is not None:
        if prior['input_fingerprint'] != bound: raise RuntimeError('Cached model inputs changed')
        return prior
    result, _ = model_cycle(data, selected, evaluation, features, seed, config, session, metadata)
    return commit(dest, {'input_fingerprint': bound, **result})


def verify_decision_lock(root, decisions=None):
    root = Path(root); lock = cached(root / 'decision_lock.json')
    if lock is None: raise RuntimeError('Outer evaluation requires a committed decision lock')
    for rel, sha in lock['source_sha256'].items():
        if digest(root / rel) != sha: raise RuntimeError('Locked candidate/decision source changed')
    current = cached(root / 'decisions.json')
    if current is None or tuple(current['selections']) != ARMS or current.get('reference_used_in_decisions') is not False:
        raise RuntimeError('Invalid four-arm decision lock')
    if decisions is not None and current != serial(decisions): raise RuntimeError('Decision object differs from locked bytes')
    return lock


def run_fold(config, paths, partitions, fold, role, session, *, output_root=None):
    session.require_execution()
    if role not in ('main', 'replay') or session.role != role: raise PermissionError('Wrong execution role')
    if session.task != f'fold-{fold}': raise PermissionError('Authorized task and requested outer fold differ')
    if role == 'replay' and fold != 0: raise ValueError('Only complete fold-0 replay is frozen')
    validate_config(config); root = _root(session, output_root)
    data = load_data(config, paths)
    outer = next(r for r in partitions['outer_folds'] if r['fold'] == fold)
    test, pool, original = (positions(outer[k]) for k in ('test', 'pool', 'original_train'))
    if len(pool) != 8000: raise ValueError('Frozen development pool must contain exactly 8000 rows')
    assert_pair(data, pool, test); assert_pair(data, original, test)
    if not set(pool) <= set(original): raise ValueError('Development pool escaped original outer train')
    bundle_bound = fingerprint({'config': config, 'outer': outer, 'contexts': {k: partitions['contexts'][k]
                                 for k in outer['inner_context_ids'] + [outer['refit_context_id']]}})
    previous = cached(root / 'bundle.json')
    if previous is not None:
        if previous['input_fingerprint'] != bundle_bound: raise RuntimeError('Completed bundle changed')
        for rel, sha in previous['artifact_sha256'].items():
            if digest(root / rel) != sha: raise RuntimeError('Completed bundle artifact changed')
        verify_decision_lock(root)
        return previous
    evaluations = []
    for i, key in enumerate(outer['inner_context_ids']):
        context = partitions['contexts'][key]
        train, validation = positions(context['train']), positions(context['validation'])
        assert_pair(data, train, validation)
        if set(train) | set(validation) != set(pool): raise ValueError('Inner folds do not cover pool')
        selectors = {m: build_selector(data, context, m, config, session, root / f'inner-{i}' / m) for m in METHODS}
        rows = []
        for cid, (method, k) in enumerate((m, k) for m in METHODS for k in range(1, 13)):
            features = selectors[method]['paths'][k - 1]
            mask = sum(1 << list(data.X.columns).index(f) for f in features)
            model_path = root / f'inner-{i}' / method / 'models' / f'mask-{mask}.json'
            model = _model(data, selectors[method]['selected_positions'], validation, features,
                outer['inner_model_seeds'][i], config, session, model_path,
                {'phase': 'candidate', 'fold': fold, 'inner': i, 'method': method, 'mask': mask})
            rows.append({'candidate_id': cid, 'method': method, 'k': k, 'features': features, 'mask': mask,
                         'f1': model['metrics']['f1'], 'actual_rho': selectors[method]['actual_rho'],
                         'model_path': model_path.relative_to(root).as_posix(), 'model_sha256': digest(model_path)})
        commit(root / f'inner-{i}' / 'candidates.json', rows); evaluations.append(rows)
    decisions = commit(root / 'decisions.json', make_four_decisions(evaluations, config))
    lock_sources = ['decisions.json'] + [f'inner-{i}/candidates.json' for i in range(3)]
    lock = commit(root / 'decision_lock.json', {'status': 'locked_before_any_outer_prediction',
                  'fold': fold, 'outer_test_positions_sha256': fingerprint(test.tolist()),
                  'source_sha256': {rel: digest(root / rel) for rel in lock_sources},
                  'reference_used_in_decisions': False, 'scientific_identity': bundle_bound})
    verify_decision_lock(root, decisions)
    context = partitions['contexts'][outer['refit_context_id']]
    selectors = {m: build_selector(data, context, m, config, session, root / 'refit' / m) for m in METHODS}
    outer_models = {}
    for arm, candidate in decisions['selections'].items():
        verify_decision_lock(root, decisions); session.checkpoint()
        selector = selectors[candidate['method']]; features = selector['paths'][candidate['k'] - 1]
        mask = sum(1 << list(data.X.columns).index(f) for f in features)
        path = root / 'outer_models' / candidate['method'] / f'mask-{mask}.json'
        model = _model(data, selector['selected_positions'], test, features, outer['outer_model_seed'],
            config, session, path, {'phase': 'final', 'fold': fold, 'arm': arm, 'mask': mask})
        n = len(selector['selected_positions']); d = len(data.X.columns)
        outer_models[arm] = {'arm': arm, 'candidate': candidate, 'features': features, 'metrics': model['metrics'],
            'selected_rows': n, 'pool_rows': len(pool), 'original_outer_train_rows': len(original),
            'rho_pool': n / len(pool), 'rho_original_outer_train': n / len(original),
            'cell_fraction_pool': n / len(pool) * len(features) / d,
            'cell_fraction_original_outer_train': n / len(original) * len(features) / d,
            'model_path': path.relative_to(root).as_posix(), 'model_sha256': digest(path)}
    verify_decision_lock(root, decisions)
    reference_path = root / 'outer_models' / 'complete_outer_training_reference.json'
    reference = _model(data, original, test, tuple(data.X.columns), outer['reference_model_seed'], config,
        session, reference_path, {'phase': 'reference', 'fold': fold, 'outside_four_arm_decisions': True})
    pool_f1 = outer_models['full']['metrics']['f1']; reference_f1 = reference['metrics']['f1']
    for arm in outer_models.values():
        arm['loss_vs_pool_full'] = pool_f1 - arm['metrics']['f1']
        arm['loss_vs_complete_outer_training_reference'] = reference_f1 - arm['metrics']['f1']
        arm['pool_tolerance_violation'] = arm['loss_vs_pool_full'] > config['epsilon'] + config['numeric_tolerance']
    # Public outer result release is atomic after every arm/reference is ready.
    released = commit(root / 'outer_release.json', {'status': 'all_four_arms_and_reference_released_together',
        'decision_lock_sha256': digest(root / 'decision_lock.json'), 'arms': outer_models,
        'complete_outer_training_reference': {'metrics': reference['metrics'], 'model_sha256': digest(reference_path),
            'model_path': reference_path.relative_to(root).as_posix(), 'rows': len(original), 'used_in_decisions': False},
        'pool_full_loss_vs_complete_outer_training': reference_f1 - pool_f1,
        'outer_test_positions': test.tolist(), 'official_adult_test_used': False})
    all_files = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*.json') if p.name != 'bundle.json'}
    return commit(root / 'bundle.json', {'input_fingerprint': bundle_bound, 'status': 'complete_pending_independent_audit',
        'fold': fold, 'role': role, 'arms': list(ARMS), 'reference_count': 1, 'artifact_sha256': all_files,
        'released_results_sha256': digest(root / 'outer_release.json'), 'automatic_resume': False})
