"""Future 75-task training-only timing probe and pure projection arithmetic."""
from contextlib import contextmanager
import math
import time
from pathlib import Path
from adapter import validate_config
from projection import projection_terms, total_seconds
from private_science.engine import (load_data, positions, fingerprint, cached, commit, read, sample_rows,
                                   redundancy, UtilityGame, model_cycle, assert_pair)


class TimingSession:
    """Meter the exact existing session; this wrapper grants no authority."""
    def __init__(self, session):
        self.parent = session; self.role = session.role; self.task = session.task
        self.counts = {'classifier': 0, 'encoder': 0, 'kmeans': 0}
        self.operation_seconds = {k: [] for k in self.counts}
    def require_execution(self): return self.parent.require_execution()
    def checkpoint(self): return self.parent.checkpoint()
    def artifact_path(self, relative): return self.parent.artifact_path(relative)
    def bind_classifier(self, selected, evaluation, features, metadata):
        return self.parent.bind_classifier(selected, evaluation, features, metadata)
    def record_event(self, name, metadata, seconds):
        return self.parent.record_event(name, metadata, seconds)
    @contextmanager
    def operation(self, kind, metadata):
        begin = time.perf_counter()
        with self.parent.operation(kind, metadata) as operation_id:
            self.counts[kind] += 1
            yield operation_id
        self.operation_seconds[kind].append(time.perf_counter() - begin)


def _maximum(values): return max(values, default=0.)


def run_task(config, paths, partitions, task, session):
    session.require_execution(); validate_config(config)
    if session.role != 'probe' or session.task != task['task_id']:
        raise PermissionError('Timing task must match authorized probe session')
    frozen_tasks = read(paths['timing_tasks_file'])['tasks']
    matched = [r for r in frozen_tasks if r['task_id'] == task['task_id']]
    if len(matched) != 1 or fingerprint(matched[0]) != fingerprint(task):
        raise PermissionError('Timing task differs from frozen manifest')
    path = Path(session.artifact_path('timing.json'))
    if path.exists():
        cached(path)
        raise RuntimeError('Completed timing task is immutable; do not reuse or repeat it as a new measurement')
    data = load_data(config, paths)
    outer = next(r for r in partitions['outer_folds'] if r['fold'] == task['fold'])
    original = set(positions(outer['original_train']).tolist()); forbidden = set(positions(outer['test']).tolist())
    train, prediction = positions(task['train_positions']), positions(task['prediction_positions'])
    if not set(train) <= original or not set(prediction) <= original or (set(train) | set(prediction)) & forbidden:
        raise ValueError('Timing may only touch its own outer-training rows')
    metered = TimingSession(session); begin = time.perf_counter(); cpu = time.process_time()
    maxima = {k: 0. for k in ('utility_cycle', 'full_candidate_cycle', 'reference_cycle', 'encoder',
                             'kmeans', 'game_setup', 'selector_setup')}
    cycles, details = [], {}
    if task['kind'] == 'selector_timing':
        if task['prediction_overlap_allowed']: raise ValueError('Selector timing validation cannot overlap')
        assert_pair(data, train, prediction)
        context = partitions['contexts'][task['partition_id']]
        if train.tolist() != context['train']['positions']: raise ValueError('Timing context identity mismatch')
        masks = task['masks']
        if len(masks) != 21 or len(set(masks)) != 21 or any(not 0 < m < 4096 for m in masks):
            raise ValueError('Exactly 21 distinct frozen original-variable masks required')
        setup = time.perf_counter()
        selected, sampler = sample_rows(data, train, task['method'], task['selector_sampler_seed'], config,
                                        metered, {'phase': 'selector_setup', 'fold': task['fold'], 'timing_only': True})
        _, redundancy_record = redundancy(data, selected, metered,
                                           {'phase': 'selector_setup', 'fold': task['fold'], 'timing_only': True})
        maxima['selector_setup'] = time.perf_counter() - setup
        record = next(r for r in context['games'] if r['resample_index'] == task['resample_index'])
        if record['game_seed'] != task['game_seed']: raise ValueError('Timing game seed mismatch')
        game = UtilityGame(data, record, task['method'], config, metered,
                           {'phase': 'utility', 'fold': task['fold'], 'timing_only': True})
        for mask in masks:
            metered.checkpoint()
            game.utility(tuple(f for i, f in enumerate(data.X.columns) if mask & (1 << i)))
        cycles = game.cycles
        maxima['utility_cycle'] = _maximum(r['cycle_seconds'] for r in cycles)
        maxima['game_setup'] = game.initialization_seconds
        candidate, _ = model_cycle(data, selected, prediction, tuple(data.X.columns), task['candidate_model_seed'],
            config, metered, {'phase': 'candidate', 'fold': task['fold'], 'timing_only': True}, keep_predictions=False)
        maxima['full_candidate_cycle'] = candidate['cycle_seconds']
        expected = {'classifier': 64, 'encoder': 69 if task['method'] == 'robust_half' else 65,
                    'kmeans': 8 if task['method'] == 'robust_half' else 0}
        details = {'masks': masks, 'utility_fold_cycle_measurements': cycles,
                   'candidate_cycle': {k: candidate[k] for k in ('cycle_seconds', 'encoder_seconds', 'classifier_seconds', 'iterations')},
                   'training_rows': len(train), 'selected_rows': len(selected), 'prediction_rows': len(prediction),
                   'prediction_overlap_allowed': False,
                   'candidate_encoded_d': candidate['encoder']['encoded_d'],
                   'utility_scores_retained': False, 'predictions_retained': False}
    elif task['kind'] == 'full_reference_timing':
        if set(train) != original or task['prediction_overlap_allowed'] is not True:
            raise ValueError('Reference timing must declare complete original training and timing-only overlap')
        reference, _ = model_cycle(data, train, prediction, tuple(data.X.columns), task['candidate_model_seed'],
            config, metered, {'phase': 'reference_timing', 'fold': task['fold'], 'timing_only': True},
            allow_timing_overlap=True, keep_predictions=False)
        maxima['reference_cycle'] = reference['cycle_seconds']
        expected = {'classifier': 1, 'encoder': 1, 'kmeans': 0}
        details = {'training_rows': len(train), 'prediction_rows': len(prediction), 'prediction_overlap_allowed': True,
                   'overlap_purpose': 'training-only transformation/prediction timing; no validation or predictive evidence',
                   'encoded_d': reference['encoder']['encoded_d'], 'iterations': reference['iterations'],
                   'utility_scores_retained': False, 'predictions_retained': False}
    else: raise ValueError('Unknown frozen timing task kind')
    if metered.counts != expected: raise RuntimeError('Timing operation counts disagree with frozen task contract')
    maxima['encoder'] = _maximum(metered.operation_seconds['encoder'])
    maxima['kmeans'] = _maximum(metered.operation_seconds['kmeans'])
    session.checkpoint()
    result = {'status': 'complete', 'task_id': task['task_id'], 'task_fingerprint': fingerprint(task),
        'kind': task['kind'], 'fold': task['fold'], 'counts': metered.counts, 'maxima': maxima,
        'actual_wall_seconds': time.perf_counter() - begin, 'cpu_seconds': time.process_time() - cpu,
        'cache_reuse': False, 'outer_test_used': False, 'official_adult_test_used': False,
        'details': details, 'operation_seconds': metered.operation_seconds,
        'OS_cache_flushed': False, 'can_issue_pass_gate': False}
    return commit(path, result)


def summarize_timing(records, config, *, probe_actual_seconds=None, old_probe_seconds=None):
    """Pure arithmetic; a result here is never an execution authorization."""
    validate_config(config)
    expected_ids = {f'fold{f}-{context}-{method}-repeat{r}' for f in range(5)
                    for context in ('inner', 'refit') for method in ('full', 'robust_half') for r in range(3)}
    expected_ids |= {f'fold{f}-full-reference-repeat{r}' for f in range(5) for r in range(3)}
    ids = [r.get('task_id') for r in records]
    if len(records) != 75 or len(set(ids)) != 75 or set(ids) != expected_ids:
        raise ValueError('All 75 unique prespecified timing records are required')
    if any(r.get('status') != 'complete' or r.get('cache_reuse') is not False or r.get('outer_test_used') is not False for r in records):
        raise ValueError('Incomplete, cached, or outer-test timing result')
    totals = {k: sum(r['counts'][k] for r in records) for k in ('classifier', 'encoder', 'kmeans')}
    if totals != {'classifier': 3855, 'encoder': 4035, 'kmeans': 240}:
        raise ValueError('Successful timing counts do not reconcile')
    keys = ('utility_cycle', 'full_candidate_cycle', 'reference_cycle', 'encoder', 'kmeans', 'game_setup', 'selector_setup')
    maxima = {k: max(float(r['maxima'][k]) for r in records) for k in keys}
    if any(not math.isfinite(v) or v <= 0 for v in maxima.values()): raise ValueError('Missing or invalid measured timing maximum')
    measured_record_sum = sum(r['actual_wall_seconds'] for r in records)
    if probe_actual_seconds is None or old_probe_seconds is None:
        return {'status': 'incomplete_ledger_wall_time_required', 'maxima': maxima,
                'successful_record_wall_seconds': measured_record_sum, 'can_authorize_execution': False}
    if type(probe_actual_seconds) not in (int,float) or not math.isfinite(probe_actual_seconds) or probe_actual_seconds < measured_record_sum:
        raise ValueError('Ledger probe wall time must cover every successful record plus failed/interrupted work')
    if type(old_probe_seconds) not in (int,float) or not math.isfinite(old_probe_seconds) or old_probe_seconds < 0:
        raise ValueError('Preserved old probe wall time must be supplied separately')
    u, c, ref = maxima['utility_cycle'], maxima['full_candidate_cycle'], maxima['reference_cycle']
    rates = {'t_utility':u,'t_full_candidate':c,'t_reference':ref,
             't_encoder':maxima['encoder'],'t_kmeans':maxima['kmeans'],
             't_game_setup':maxima['game_setup'],'t_selector_setup':maxima['selector_setup']}
    terms = projection_terms(rates)
    predicted = total_seconds(rates, old_probe_seconds, probe_actual_seconds)
    within = predicted <= 172800 and probe_actual_seconds <= 7200
    return {'status': 'candidate_pass_pending_independent_review' if within else 'failed_runtime_feasibility',
        'maxima': maxima, 'terms_before_safety_multiplier': terms,
        'probe_actual_seconds': old_probe_seconds + probe_actual_seconds,
        'old_probe_seconds': old_probe_seconds, 'new_probe_seconds': probe_actual_seconds,
        'measured_parameters': {'t_utility': u, 't_full_candidate': c, 't_reference': ref,
            't_encoder': maxima['encoder'], 't_kmeans': maxima['kmeans'],
            't_game_setup': maxima['game_setup'], 't_selector_setup': maxima['selector_setup'],
            'T_probe_actual': old_probe_seconds + probe_actual_seconds},
        'successful_record_wall_seconds': measured_record_sum, 'predicted_seconds': predicted,
        'warning_seconds': 172800, 'within_48_hours': within, 'successful_counts': totals,
        'engineering_allowance_not_absolute_or_statistical_bound': True,
        'source_records_require_independent_hash_and_ledger_verification': True,
        'can_authorize_execution': False, 'automatic_resume': False}
