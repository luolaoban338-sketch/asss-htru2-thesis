"""Instrumented native PowerSHAP selector on one bound outer-training fold."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

from .common import (ROOT, HERE, QA, RUN, read, write, sha, array_sha,
                     verify_freeze, cached, unit_context)


def row_ids_sha(row_ids):
    return hashlib.sha256(json.dumps(row_ids, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def json_numeric_matrix(values):
    """Lossless nonfinite labels keep failure evidence valid strict JSON."""
    return [[float(value) if np.isfinite(value) else
             ('NaN' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity')
             for value in row] for row in np.asarray(values)]


def load_train(fold, spec):
    """Read only the specified training rows into selector arrays."""
    data = spec['data']
    for key in ('data_source', 'outer_partitions'):
        assert sha(data[key]['path']) == data[key]['sha256'], key
    splits = read(data['outer_partitions']['path'])['folds']
    assert fold in spec['outer_fold_order'] and 0 <= fold < len(splits)
    split = splits[fold]
    positions = split['train_positions']
    test = set(split['test_positions'])
    expected = next(v for v in data['folds'] if v['fold'] == fold)
    assert len(positions) == expected['train_n']
    assert len(set(positions)) == len(positions) and not set(positions) & test
    assert len(test) == expected['test_n']
    assert len(positions) + len(test) == data['n']
    selected = set(positions)
    rows = {}
    with open(data['data_source']['path'], encoding='utf-8-sig', newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i in selected:
                rows[i] = row
    x = np.asarray([[float(rows[i][k]) for k in data['predictors']]
                    for i in positions], dtype=np.float64)
    y = np.asarray([int(rows[i]['target']) for i in positions], dtype=np.int64)
    assert x.shape == (expected['train_n'], data['d'])
    assert np.isfinite(x).all() and set(y) == {0, 1}
    row_ids = [rows[i]['row_id'] for i in positions]
    assert row_ids == split['train_row_ids'], 'Frozen row identities differ'
    return x, y, np.asarray(positions, dtype=np.int64), row_ids


def counted_fit(model, X, Y, args, kwargs, context, model_params,
                ledger, unit, deadline, original_fit):
    rec = context['fit_context']
    assert rec is not None, 'Unbound fit'
    assert not args and set(kwargs) == {'eval_set'}, 'Unexpected fit kwargs'
    assert time.time() < deadline, 'Deadline reached'
    assert not (HERE/'STOP_REQUESTED').exists() and not (RUN/'STOP_REQUESTED').exists(), 'STOP set'
    assert rec['calls'] == 0, 'Duplicate internal fit'
    rec['calls'] += 1
    assert array_sha(X) == rec['X_train_sha256'] and array_sha(Y) == rec['Y_train_sha256']
    assert array_sha(kwargs['eval_set'][0]) == rec['X_val_sha256']
    assert array_sha(kwargs['eval_set'][1]) == rec['Y_val_sha256']
    expected = {**model_params, 'random_seed': rec['seed']}
    assert model.get_params() == expected, model.get_params()
    attempt = ledger.reserve(unit, 'catboost', rec['seed'])
    context['fit_calls'] += 1
    start = time.perf_counter()
    try:
        value = original_fit(model, X, Y, **kwargs)
    except BaseException as exc:
        ledger.finish(attempt, 'failed', time.perf_counter()-start, repr(exc))
        raise
    seconds = time.perf_counter()-start
    ledger.finish(attempt, 'succeeded', seconds)
    rec.update(attempt_id=attempt, specified_params=model.get_params(),
               effective_params=model.get_all_params(), tree_count=int(model.tree_count_),
               fit_seconds=seconds)
    return value


def native_validity(stats, selected, iterations, warning_records, spec):
    """Validate native termination without altering native statistics or selection."""
    rules = spec['selector']['statistical_validity']
    alpha = spec['selector']['power_alpha']
    required_power = spec['selector']['power_req_iterations']
    power_column = 'power_0.01_alpha'
    iterations_column = '0.99_power_its_req'
    assert set(rules['finite_required']).issubset(stats.columns)
    finite = bool(np.isfinite(stats[rules['finite_required']].values.astype(float)).all())
    probabilities = bool(stats['p_value'].between(0, 1).all()
                         and stats[power_column].between(0, 1).all())
    original_indices = [i for i in stats.index if isinstance(i, (int, np.integer))]
    assert sorted(int(i) for i in original_indices) == list(range(spec['data']['d']))
    expected_selected = [int(i) for i in original_indices if stats.loc[i, 'p_value'] < alpha]
    assert selected == sorted(expected_selected), 'Support does not match native p-value rule'
    significant = stats.loc[expected_selected]
    power = bool((significant[power_column] >= required_power-rules['absolute_tolerance']).all())
    requirement = bool((np.ceil(significant[iterations_column]) <= iterations).all())
    warning_valid = not any(w['category'] in ('ConvergenceWarning', 'RuntimeWarning')
                            for w in warning_records)
    validation = dict(finite=finite, probabilities=probabilities, power=power,
                      required_iterations=requirement, warnings=warning_valid)
    statistics_valid = finite and probabilities and warning_valid
    complete = statistics_valid and power and requirement
    reason = ('no_significant_features' if complete and not selected else
              'power_satisfied' if complete else
              'invalid_native_statistics' if not statistics_valid else
              'incomplete_native_power_requirement')
    return complete, statistics_valid, validation, reason


def execute(fold, replay, quota_path):
    assert fold in range(5)
    assert not replay or fold == 0, 'Only formal fold 0 has a cold replay'
    unit = f"native_{'replay' if replay else 'main'}_f{fold}"
    prior = cached(unit, 'native')
    if prior is not None:
        return prior
    with unit_context(unit, quota_path, 'native') as ctx:
        tick = time.perf_counter()
        out = ctx.out
        spec = read(HERE/'frozen_spec.json')
        if replay:
            assert cached('native_main_f0', 'native') is not None, 'Formal main fold 0 must complete first'
        # Library initialization is retained in operational time and outside selector time.
        from catboost import CatBoostClassifier
        from powershap import PowerShap
        from powershap.shap_wrappers.shap_explainer import ShapExplainer, CatboostExplainer
        from sklearn.model_selection import train_test_split

        x, y, positions, row_ids = load_train(fold, spec)
        write(out/'training_binding.json', dict(
            outer_fold=fold, train_positions=positions.tolist(), row_ids=row_ids,
            train_row_ids_sha256=row_ids_sha(row_ids),
            x_sha256=array_sha(x), y_sha256=array_sha(y), n=len(x), d=x.shape[1],
            outer_test_predictions=0))
        model_keys = ('iterations', 'depth', 'learning_rate', 'loss_function', 'od_type',
                      'od_wait', 'use_best_model', 'thread_count', 'task_type', 'verbose',
                      'allow_writing_files')
        model_params = {k: spec['internal_model'][k] for k in model_keys}
        selector_keys = ('power_iterations', 'power_alpha', 'val_size', 'power_req_iterations',
                         'include_all', 'automatic', 'force_convergence', 'limit_convergence_its',
                         'limit_automatic', 'limit_incremental_iterations', 'limit_recursive_automatic',
                         'stratify', 'cv', 'show_progress', 'verbose')
        selector_params = {k: spec['selector'][k] for k in selector_keys}
        context = dict(records=[], batches=[], fit_context=None, batch=None, fit_calls=0)
        original_fit = CatBoostClassifier.fit
        original_explain = ShapExplainer.explain
        original_get = CatboostExplainer._fit_get_shap

        def safe_fit(model, X, Y, *args, **kwargs):
            return counted_fit(model, X, Y, args, kwargs, context, model_params,
                               ctx.ledger, unit, ctx.deadline, original_fit)

        def instrument_get(explainer, X_train, Y_train, X_val, Y_val, random_seed, **kwargs):
            start = time.perf_counter()
            batch = context['batch']
            local = len(context['records'])-batch['first_iteration']
            assert not kwargs and random_seed == batch['random_seed_start']+local
            train, val = train_test_split(np.arange(len(x)), test_size=selector_params['val_size'],
                                         random_state=local, stratify=None)
            train, val = np.sort(train), np.sort(val)
            augmented = np.column_stack([x, np.random.RandomState(random_seed).uniform(-1, 1, len(x))])
            assert np.array_equal(X_train, augmented[train]) and np.array_equal(X_val, augmented[val])
            assert np.array_equal(Y_train, y[train]) and np.array_equal(Y_val, y[val])
            rec = dict(iteration=len(context['records']), seed=int(random_seed),
                       batch=len(context['batches']), local_iteration=local,
                       train_positions_sha256=array_sha(positions[train]),
                       val_positions_sha256=array_sha(positions[val]), train_n=len(train), val_n=len(val),
                       X_train_sha256=array_sha(X_train), Y_train_sha256=array_sha(Y_train),
                       X_val_sha256=array_sha(X_val), Y_val_sha256=array_sha(Y_val), calls=0)
            context['fit_context'] = rec
            try:
                raw = original_get(explainer, X_train, Y_train, X_val, Y_val, random_seed, **kwargs)
            finally:
                context['fit_context'] = None
            assert rec['calls'] == 1
            dest = out/'iterations'/f"{rec['iteration']:03d}"
            dest.mkdir(parents=True, exist_ok=False)
            np.save(dest/'raw_shap.npy', raw, allow_pickle=False)
            rec['raw_shap_sha256'] = sha(dest/'raw_shap.npy')
            rec['iteration_seconds'] = time.perf_counter()-start
            write(dest/'record.json', rec)
            context['records'].append(rec)
            return raw

        def instrument_explain(explainer, *args, **kwargs):
            assert not args and kwargs.get('groups') is None
            assert kwargs.get('cv_split') is None and kwargs.get('stratify') is None
            batch = dict(first_iteration=len(context['records']), loop_its=int(kwargs['loop_its']),
                         random_seed_start=int(kwargs.get('random_seed_start', 0)))
            assert 0 < batch['loop_its'] <= 10 and batch['first_iteration']+batch['loop_its'] <= 40
            assert batch['random_seed_start'] == batch['first_iteration'], 'Native iteration seed drift'
            context['batch'] = batch
            start = time.perf_counter()
            values = original_explain(explainer, **kwargs)
            batch.update(seconds=time.perf_counter()-start, columns=[str(k) for k in values.columns],
                         mean_values=json_numeric_matrix(values.values))
            write(out/'batches'/f"{len(context['batches']):02d}.json", batch)
            context['batches'].append(batch)
            return values

        CatBoostClassifier.fit = safe_fit
        ShapExplainer.explain = instrument_explain
        CatboostExplainer._fit_get_shap = instrument_get
        initialized = time.perf_counter()-tick
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                selector = PowerShap(model=CatBoostClassifier(**model_params), **selector_params)
                start = time.perf_counter()
                selector.fit(x, y, groups=None)
                selection_seconds = time.perf_counter()-start
            warning_records = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
        finally:
            CatBoostClassifier.fit = original_fit
            ShapExplainer.explain = original_explain
            CatboostExplainer._fit_get_shap = original_get
        stats = selector._processed_shaps_df
        values = stats.values.astype(float)
        selected = [int(i) for i in np.flatnonzero(selector.get_support())]
        iterations = len(context['records'])
        assert context['fit_calls'] == iterations and 0 < iterations <= 40
        # Save native statistics before validation, including invalid nonfinite output.
        np.save(out/'stats_raw.npy', values, allow_pickle=False)
        write(out/'stats.json', dict(index=[str(i) for i in stats.index], columns=list(stats.columns),
                                    values=json_numeric_matrix(values), selected_indices=selected))
        write(out/'warnings.json', warning_records)
        complete, statistics_valid, validation, reason = native_validity(
            stats, selected, iterations, warning_records, spec)
        selection = dict(
            plan_id=spec['plan_id'], plan_spec_sha256=sha(HERE/'frozen_spec.json'),
            execution_manifest_sha256=sha(HERE/'execution_manifest.json'),
            native_environment_sha256=sha(spec['environments']['native_selector']['record']),
            data_sha256=spec['data']['data_source']['sha256'],
            split_sha256=spec['data']['outer_partitions']['sha256'], outer_fold=fold,
            train_row_ids_sha256=row_ids_sha(row_ids), ordered_feature_names=spec['data']['predictors'],
            selected_indices=selected, native_complete=complete, statistics_valid=statistics_valid,
            actual_iterations=iterations)
        write(out/'selection.json', selection)
        iteration_seconds = sum(r['iteration_seconds'] for r in context['records'])
        result = dict(unit=unit, status='completed' if complete else reason, outer_fold=fold,
                      native_complete=complete, statistics_valid=statistics_valid,
                      native_outcome=reason, validation=validation,
                      iterations=iterations, fit_calls=context['fit_calls'], batches=len(context['batches']),
                      selected_indices=selected, selection=selection,
                      initialization_seconds=initialized, selector_seconds=selection_seconds,
                      iteration_seconds_sum=iteration_seconds,
                      slowest_iteration_seconds=max(r['iteration_seconds'] for r in context['records']),
                      non_iteration_seconds=max(0, selection_seconds-iteration_seconds),
                      physical_catboost_tree_count=sum(r['tree_count'] for r in context['records']),
                      execution_manifest_sha256=sha(HERE/'execution_manifest.json'),
                      outer_test_predictions=0, encoder_fits=0, kmeans_fits=0)
        if not complete:
            result['payload_hashes'] = {p.relative_to(out).as_posix(): sha(p)
                                       for p in out.rglob('*') if p.is_file()}
            write(out/'result.json', result)
            raise RuntimeError(reason)
        ctx.result(result)
        print(json.dumps({k: result[k] for k in ('unit', 'outer_fold', 'native_complete',
                                                'iterations', 'fit_calls', 'selected_indices',
                                                'selector_seconds')}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--quota', type=Path)
    args = parser.parse_args()
    execute(args.fold, args.replay, args.quota)


if __name__ == '__main__':
    main()
