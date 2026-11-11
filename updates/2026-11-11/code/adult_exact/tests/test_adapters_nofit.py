"""No-fit contracts. All real estimator fit/predict entries are hard sentinels.

Only explicitly named test doubles produce synthetic outputs. This module never
uses real Adult data to fit, transform or predict any model.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
import adapter
import audit
import timing
from private_science import engine
from private_science.preprocessing import TabularEncoder


class TestDoubleSession:
    """An explicitly fake session: production fit/predict remain patched shut."""
    __test__ = False
    def __init__(self, root, role='main', task='fold-0'):
        self.root, self.role, self.task = Path(root), role, task
        self.operations = []; self.events = []; self.authorized = True
    def require_execution(self):
        if not self.authorized: raise PermissionError('Denied fake session')
    def checkpoint(self): self.require_execution()
    def artifact_path(self, relative): return self.root / relative
    def record_event(self, name, metadata, seconds): self.events.append((name, metadata, seconds))
    def bind_classifier(self, selected, evaluation, features, metadata):
        return {**metadata,'classifier_bucket':'reference' if 'reference' in metadata.get('phase','') else 'pool'}
    @contextmanager
    def operation(self, kind, metadata):
        self.require_execution(); self.operations.append((kind, metadata)); yield len(self.operations)


class TestDoubleEstimator:
    __test__ = False
    def fit(self, X, y=None): self.fit_seen = True; return self


def config(): return json.loads((STAGE / 'config.json').read_text('utf-8-sig'))


class NoFitContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.real_calls = []
        def forbidden(*args, **kwargs):
            self.real_calls.append('attempt'); raise AssertionError('REAL FIT/PREDICT FORBIDDEN IN NO-FIT TEST')
        for cls, names in [(LogisticRegression, ('fit', 'predict', 'predict_proba')),
                           (RandomForestClassifier, ('fit', 'predict', 'predict_proba')),
                           (KMeans, ('fit', 'fit_predict', 'predict')),
                           (TabularEncoder, ('fit', 'fit_transform')),
                           (audit.AuditEncoder, ('fit',))]:
            for name in names: self.stack.enter_context(patch.object(cls, name, forbidden))
        root = os.environ.get('ADULT_NOFIT_TEST_TMP')
        if not root: raise RuntimeError('Set ADULT_NOFIT_TEST_TMP to approved F scratch storage')
        Path(root).mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='adapter_contract_', dir=root)
        self.addCleanup(self.temp.cleanup); self.addCleanup(self.stack.close)
        self.addCleanup(lambda: self.assertEqual(self.real_calls, [], 'No test may reach a real estimator entry'))
        self.root = Path(self.temp.name); self.session = TestDoubleSession(self.root)
        self.cfg = config()

    def tables(self):
        return [[{'candidate_id': i, 'method': adapter.METHODS[i // 12], 'k': i % 12 + 1,
                  'f1': .8, 'actual_rho': 1. if i < 12 else .5} for i in range(24)] for _ in range(3)]

    def test_exact_four_arm_scope(self):
        result = adapter.make_four_decisions(self.tables(), self.cfg)
        self.assertEqual(tuple(result['selections']), adapter.ARMS)
        self.assertFalse(result['reference_used_in_decisions'])
        self.assertEqual(result['selections']['joint']['method'], 'robust_half')
        self.assertEqual(result['selections']['joint']['k'], 1)

    def test_fallback_is_explicit(self):
        tables = self.tables()
        for table in tables:
            for row in table: row['f1'] = .8 if row['method'] == 'full' else .7
        out = adapter.make_four_decisions(tables, self.cfg)
        self.assertEqual(out['selections']['sample_only']['method'], 'full')

    def test_bad_candidate_tables_are_rejected(self):
        for change in ('duplicate', 'extra', 'nan', 'inf'):
            tables = self.tables()
            if change == 'duplicate': tables[0][1]['candidate_id'] = 0
            if change == 'extra': tables[0].append(dict(tables[0][0]))
            if change == 'nan': tables[0][0]['actual_rho'] = float('nan')
            if change == 'inf': tables[0][0]['f1'] = float('inf')
            with self.subTest(change=change), self.assertRaises(ValueError): adapter.make_four_decisions(tables, self.cfg)

    def test_unknown_dimension_and_grid_are_rejected(self):
        with self.assertRaises(ValueError): adapter.make_four_decisions(self.tables(), self.cfg, 11)
        changed = deepcopy(self.cfg); changed['M'] = 32
        with self.assertRaises(ValueError): adapter.validate_config(changed)

    def test_task_fold_binding_precedes_data_access(self):
        with patch.object(adapter, 'load_data', side_effect=AssertionError('Data should not load')):
            with self.assertRaises(PermissionError): adapter.run_fold(self.cfg, {}, {}, 1, 'main', self.session)

    def test_denied_session_precedes_all_work(self):
        self.session.authorized = False
        with self.assertRaises(PermissionError): adapter.run_fold(self.cfg, {}, {}, 0, 'main', self.session)

    def test_output_root_cannot_escape(self):
        with self.assertRaises(PermissionError): adapter._root(self.session, self.root / 'elsewhere')

    def test_partial_bytes_are_never_replaced(self):
        path = self.root / 'artifact.json'; partial = self.root / 'artifact.json.partial'
        partial.write_bytes(b'interrupted-evidence')
        with self.assertRaises(RuntimeError): engine.commit(path, {'new': True})
        self.assertEqual(partial.read_bytes(), b'interrupted-evidence')
        self.assertFalse(path.exists())

    def test_completed_artifact_integrity(self):
        path = self.root / 'record.json'; engine.commit(path, {'a': 1})
        self.assertEqual(engine.cached(path), {'a': 1})
        with self.assertRaises(RuntimeError): engine.commit(path, {'a': 2})
        path.write_text('{"a":3}', encoding='utf-8')
        with self.assertRaises(RuntimeError): engine.cached(path)

    def test_only_explicit_fit_double_can_cross_operation_wrapper(self):
        result, _ = engine.checked_fit(self.session, 'classifier', TestDoubleEstimator(), [[0.]], [0],
                                       metadata={'phase': 'unit_test_double', 'fold': 0})
        self.assertTrue(result.fit_seen); self.assertEqual(len(self.session.operations), 1)

    def test_utility_unknown_or_duplicate_feature_rejected_without_fit(self):
        game = object.__new__(engine.UtilityGame); game.session = self.session; game.features = ('a', 'b')
        for features in [('x',), ('a', 'a')]:
            with self.assertRaises(ValueError): game.utility(features)

    def test_independent_decision_formula_matches_frozen_rule(self):
        table = self.tables(); table[1][0]['f1'] = .799
        self.assertEqual(audit.independent_decisions(table, self.cfg), adapter.make_four_decisions(table, self.cfg))

    def test_independent_metric_hand_calculation(self):
        out = audit.independent_metrics([0, 0, 1, 1], [.1, .6, .7, .2])
        self.assertEqual(out['f1'], .5); self.assertEqual(out['accuracy'], .5)
        self.assertEqual(out['roc_auc'], .75)

    def timing_records(self):
        records = []
        for f in range(5):
            for context in ('inner', 'refit'):
                for method in ('full', 'robust_half'):
                    for r in range(3):
                        records.append({'task_id': f'fold{f}-{context}-{method}-repeat{r}', 'status': 'complete',
                            'cache_reuse': False, 'outer_test_used': False, 'actual_wall_seconds': 1.,
                            'counts': {'classifier': 64, 'encoder': 69 if method == 'robust_half' else 65, 'kmeans': 8 if method == 'robust_half' else 0},
                            'maxima': dict.fromkeys(('utility_cycle', 'full_candidate_cycle', 'reference_cycle', 'encoder', 'kmeans', 'game_setup', 'selector_setup'), .001)})
            for r in range(3):
                records.append({'task_id': f'fold{f}-full-reference-repeat{r}', 'status': 'complete',
                    'cache_reuse': False, 'outer_test_used': False, 'actual_wall_seconds': 1.,
                    'counts': {'classifier': 1, 'encoder': 1, 'kmeans': 0},
                    'maxima': dict.fromkeys(('utility_cycle', 'full_candidate_cycle', 'reference_cycle', 'encoder', 'kmeans', 'game_setup', 'selector_setup'), .001)})
        return records

    def test_timing_projection_uses_ledger_failed_time_and_every_reserve(self):
        out = timing.summarize_timing(self.timing_records(), self.cfg, probe_actual_seconds=100., old_probe_seconds=240.8535168)
        expected = 340.8535168 + 3.75 * .001 * (301350 + 1056 + 12 + 42840 + 12 + 44132 + 152 + 384 + 96) + 3600
        self.assertAlmostEqual(out['predicted_seconds'], expected)
        self.assertFalse(out['can_authorize_execution'])
        self.assertAlmostEqual(out['measured_parameters']['T_probe_actual'], 340.8535168)

    def test_timing_never_passes_without_full_ledger_wall_time(self):
        out = timing.summarize_timing(self.timing_records(), self.cfg)
        self.assertEqual(out['status'], 'incomplete_ledger_wall_time_required')
        with self.assertRaises(ValueError): timing.summarize_timing(self.timing_records(), self.cfg, probe_actual_seconds=74., old_probe_seconds=240.8535168)

    def test_missing_or_duplicate_probe_rejected(self):
        records = self.timing_records()
        with self.assertRaises(ValueError): timing.summarize_timing(records[:-1], self.cfg, probe_actual_seconds=100., old_probe_seconds=240.8535168)
        records[-1] = deepcopy(records[0])
        with self.assertRaises(ValueError): timing.summarize_timing(records, self.cfg, probe_actual_seconds=100., old_probe_seconds=240.8535168)

    def test_timing_over_48_hours_stays_failed(self):
        records = self.timing_records()
        records[0]['maxima']['utility_cycle'] = 1.
        out = timing.summarize_timing(records, self.cfg, probe_actual_seconds=100., old_probe_seconds=240.8535168)
        self.assertEqual(out['status'], 'failed_runtime_feasibility')

    def test_direct_audit_counts_and_monotone_timing_event_with_test_doubles(self):
        names = tuple(f'f{i}' for i in range(12)); n = 24
        data = engine.Data(pd.DataFrame(np.zeros((n, 12)), columns=names), np.arange(n) % 2,
            np.array([f'r{i}' for i in range(n)]), np.array([f'g{i}' for i in range(n)]),
            pd.DataFrame(index=range(n)), names[:6], names[6:], 'explicit-audit-double')
        class TestDoubleAuditEncoder:
            def __init__(self, *args): pass
            def fit(self, frame): return self
            def transform(self, frame): return np.zeros((len(frame), 1))
        class TestDoubleAuditClassifier:
            def __init__(self, **kwargs): self.classes_ = np.array([0, 1])
            def fit(self, X, y): return self
            def predict_proba(self, X):
                p = np.tile([.2, .8], (len(X) + 1) // 2)[:len(X)]
                return np.column_stack([1 - p, p])
        self.session.role = 'audit'
        with patch.object(audit, 'AuditEncoder', TestDoubleAuditEncoder), patch.object(audit, 'LogisticRegression', TestDoubleAuditClassifier):
            _, metrics, _ = audit.direct_cycle(data, list(range(12)), list(range(12, 24)), names,
                1, self.cfg, self.session, {'phase': 'audit_utility', 'fold': 0})
        self.assertEqual(metrics['f1'], 1.)
        self.assertEqual([x[0] for x in self.session.operations], ['encoder', 'classifier'])
        self.assertEqual(self.session.events[0][0], 'audit_complete_cycle')

    def test_selector_probe_contract_with_explicit_counting_doubles(self):
        names = tuple(f'f{i}' for i in range(12)); n = 40
        data = engine.Data(pd.DataFrame(np.zeros((n, 12)), columns=names), np.arange(n) % 2,
            np.array([f'r{i}' for i in range(n)]), np.array([f'g{i}' for i in range(n)]),
            pd.DataFrame(index=range(n)), names[:6], names[6:], 'explicit-probe-double')
        train, valid = list(range(24)), list(range(24, 32))
        task = {'task_id': 'fold0-inner-robust_half-repeat0', 'kind': 'selector_timing', 'fold': 0,
                'partition_id': 'inner0', 'method': 'robust_half', 'resample_index': 0,
                'masks': list(range(1, 22)), 'train_positions': train, 'prediction_positions': valid,
                'prediction_overlap_allowed': False, 'candidate_model_seed': 2,
                'selector_sampler_seed': 3, 'game_seed': 4}
        parts = {'outer_folds': [{'fold': 0, 'original_train': {'positions': list(range(32))}, 'test': {'positions': list(range(32, 40))}}],
                 'contexts': {'inner0': {'train': {'positions': train}, 'games': [{'resample_index': 0, 'game_seed': 4}]}}}
        self.session.role = 'probe'; self.session.task = task['task_id']
        def count(session, kinds):
            for kind in kinds:
                with session.operation(kind, {'phase': 'explicit_counting_test_double', 'fold': 0}): pass
        def fake_sample(data, train, method, seed, config, session, metadata):
            count(session, ['encoder', 'kmeans', 'kmeans']); return np.asarray(train[::2]), {}
        def fake_redundancy(data, selected, session, metadata):
            count(session, ['encoder']); return np.zeros((12, 12)), {}
        class TestDoubleUtilityGame:
            def __init__(self, data, record, method, cfg, session, metadata):
                self.session = session; self.cycles = []; self.initialization_seconds = .001
                for _ in range(3): count(session, ['encoder', 'kmeans', 'kmeans'])
            def utility(self, features):
                for _ in range(3):
                    count(self.session, ['encoder', 'classifier'])
                    self.cycles.append({'cycle_seconds': .001, 'encoder_seconds': .0002, 'classifier_seconds': .0003, 'iterations': [1]})
                return .5
        def fake_candidate(*args, **kwargs):
            session = args[6]; count(session, ['encoder', 'classifier'])
            return {'cycle_seconds': .001, 'encoder_seconds': .0002, 'classifier_seconds': .0003,
                    'iterations': [1], 'encoder': {'encoded_d': 12}}, .5
        with patch.object(timing, 'read', return_value={'tasks': [task]}), patch.object(timing, 'load_data', return_value=data), \
             patch.object(timing, 'sample_rows', side_effect=fake_sample), patch.object(timing, 'redundancy', side_effect=fake_redundancy), \
             patch.object(timing, 'UtilityGame', TestDoubleUtilityGame), patch.object(timing, 'model_cycle', side_effect=fake_candidate):
            result = timing.run_task(self.cfg, {'timing_tasks_file': 'explicit-fixture'}, parts, task, self.session)
            with self.assertRaises(RuntimeError): timing.run_task(self.cfg, {'timing_tasks_file': 'explicit-fixture'}, parts, task, self.session)
        self.assertEqual(result['counts'], {'classifier': 64, 'encoder': 69, 'kmeans': 8})
        self.assertEqual(len(result['details']['utility_fold_cycle_measurements']), 63)
        self.assertFalse(result['details']['predictions_retained'])

    def test_probe_task_must_match_frozen_task_bytes(self):
        self.session.role = 'probe'; self.session.task = 'example'
        task = {'task_id': 'example', 'kind': 'selector_timing', 'fold': 0}
        with patch.object(timing, 'read', return_value={'tasks': [{**task, 'changed': True}]}):
            with self.assertRaises(PermissionError): timing.run_task(self.cfg, {'timing_tasks_file': 'explicit-fixture'}, {}, task, self.session)

    def test_full_adapter_control_flow_with_explicit_test_doubles(self):
        names = tuple(f'f{i}' for i in range(12)); size = 10000
        data = engine.Data(pd.DataFrame(np.zeros((size, 12)), columns=names), np.arange(size) % 2,
            np.array([f'r{i}' for i in range(size)]), np.array([f'g{i}' for i in range(size)]),
            pd.DataFrame(index=range(size)), names[:6], names[6:], 'explicit-test-double-data')
        pool = list(range(8000)); original = list(range(9000)); test = list(range(9000, 10000))
        contexts = {}; inner_ids = []
        for i, valid in enumerate(np.array_split(np.arange(8000), 3)):
            key = f'inner-{i}'; inner_ids.append(key)
            contexts[key] = {'partition_id': key, 'fold': 0, 'train': {'positions': sorted(set(pool) - set(valid))},
                             'validation': {'positions': valid.tolist()}, 'games': []}
        contexts['refit'] = {'partition_id': 'refit', 'fold': 0, 'train': {'positions': pool}, 'games': []}
        outer = {'fold': 0, 'pool': {'positions': pool}, 'original_train': {'positions': original},
                 'test': {'positions': test}, 'inner_context_ids': inner_ids, 'refit_context_id': 'refit',
                 'inner_model_seeds': [1, 2, 3], 'outer_model_seed': 4, 'reference_model_seed': 5}
        partitions = {'outer_folds': [outer], 'contexts': contexts}; outer_calls = []
        def fake_selector(data, context, method, cfg, session, destination):
            rows = context['train']['positions']; selected = rows if method == 'full' else rows[::2]
            return {'selected_positions': selected, 'actual_rho': len(selected) / len(rows),
                    'paths': [list(names[:k]) for k in range(1, 13)]}
        def fake_cycle(data, selected, evaluation, features, seed, cfg, session, metadata):
            if metadata['phase'] in ('final', 'reference'):
                adapter.verify_decision_lock(self.root); outer_calls.append(metadata['phase'])
            return {'metrics': {'f1': 1., 'roc_auc': 1.}, 'probabilities': [float(x) for x in data.y[evaluation]],
                    'features': list(features), 'selected_positions': list(map(int, selected)),
                    'evaluation_positions': list(map(int, evaluation)), 'seed': seed}, 1.
        with patch.object(adapter, 'load_data', return_value=data), patch.object(adapter, 'build_selector', side_effect=fake_selector), patch.object(adapter, 'model_cycle', side_effect=fake_cycle):
            result = adapter.run_fold(self.cfg, {}, partitions, 0, 'main', self.session)
            calls_before = list(outer_calls)
            adapter.run_fold(self.cfg, {}, partitions, 0, 'main', self.session)
            self.assertEqual(calls_before, outer_calls, 'Complete resume must not invoke even the model double')
        self.assertEqual(result['arms'], list(adapter.ARMS)); self.assertEqual(outer_calls.count('reference'), 1)
        released = engine.cached(self.root / 'outer_release.json')
        self.assertEqual(set(released['arms']), set(adapter.ARMS))
        for record in released['arms'].values():
            self.assertAlmostEqual(record['rho_pool'], record['selected_rows'] / 8000)
            self.assertAlmostEqual(record['rho_original_outer_train'], record['selected_rows'] / 9000)
        self.assertEqual(released['complete_outer_training_reference']['rows'], 9000)


if __name__ == '__main__': unittest.main(verbosity=2)
