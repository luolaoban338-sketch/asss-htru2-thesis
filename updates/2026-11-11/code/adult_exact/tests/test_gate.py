"""Gate contracts with 75 explicit test doubles and in-memory SQLite only.

The numerical durations below are invented constants, never measurements.
No scientific data, result files, real ledger, fit, prediction or probe is used.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import copy
import hashlib
import json
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
import gate
from runtime import ExecutionDenied


def forbidden_model_call(*args, **kwargs):
    raise AssertionError('REAL FIT/PREDICTION IS FORBIDDEN IN GATE CONTRACT TESTS')


class SyntheticEvidence:
    """A complete hypothetical trace; hashes and file access are test doubles."""
    def __init__(self):
        self.root = STAGE / 'TEST_DOUBLE_ONLY_NEVER_CREATED'
        self.paths = {'run_root': str(self.root),
                      'timing_tasks_file': str(self.root / 'timing_tasks.json'),
                      'ledger_file': str(self.root / 'NEVER_CREATED.sqlite3')}
        self.tasks, self.records = [], {}
        self.missing_files, self.changed_markers = set(), set()
        for fold in range(5):
            for context in ('inner', 'refit'):
                for method in ('full', 'robust_half'):
                    for repeat in range(3):
                        self.tasks.append({'task_id': f'fold{fold}-{context}-{method}-repeat{repeat}',
                            'kind': 'selector_timing', 'method': method, 'fold': fold,
                            'masks': list(range(1, 22)), 'prediction_overlap_allowed': False,
                            'test_double_only': True})
            for repeat in range(3):
                self.tasks.append({'task_id': f'fold{fold}-full-reference-repeat{repeat}',
                    'kind': 'full_reference_timing', 'fold': fold,
                    'prediction_overlap_allowed': True, 'test_double_only': True})
        self.db = sqlite3.connect(':memory:')
        self.db.executescript('''
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE operations(id INTEGER PRIMARY KEY,role TEXT,task TEXT,kind TEXT,status TEXT,seconds REAL);
CREATE TABLE sessions(id INTEGER PRIMARY KEY,role TEXT,status TEXT,seconds REAL);
CREATE TABLE completed_tasks(role TEXT,task TEXT,PRIMARY KEY(role,task));
INSERT INTO metadata VALUES ('freeze_sha256','test-double-freeze');
INSERT INTO sessions VALUES (1,'probe','complete',100);
''')
        for task in self.tasks:
            reference = task['kind'] == 'full_reference_timing'
            robust = task.get('method') == 'robust_half'
            counts = {'classifier': 1, 'encoder': 1, 'kmeans': 0} if reference else {
                'classifier': 64, 'encoder': 69 if robust else 65, 'kmeans': 8 if robust else 0}
            samples = {kind: [({'classifier': .0005, 'encoder': .001, 'kmeans': .002}[kind])] * count
                       for kind, count in counts.items()}
            maxima = {'utility_cycle': 0. if reference else .01,
                      'full_candidate_cycle': 0. if reference else .02,
                      'reference_cycle': .03 if reference else 0.,
                      'encoder': .001, 'kmeans': .002 if robust else 0.,
                      'game_setup': 0. if reference else .004,
                      'selector_setup': 0. if reference else .005}
            details = {'prediction_overlap_allowed': reference, 'utility_scores_retained': False,
                       'predictions_retained': False, 'training_rows': 100, 'prediction_rows': 20}
            if reference:
                details.update(overlap_purpose='training-only transformation/prediction timing; no validation or predictive evidence',
                               encoded_d=12, iterations=[1])
            else:
                details.update(masks=task['masks'],
                    utility_fold_cycle_measurements=[{'cycle_seconds': .01, 'encoder_seconds': .001,
                        'classifier_seconds': .0005, 'cpu_seconds': .005, 'iterations': [1]} for _ in range(63)],
                    candidate_cycle={'cycle_seconds': .02, 'encoder_seconds': .001,
                        'classifier_seconds': .0005, 'iterations': [1]})
            self.records[task['task_id']] = {'task_id': task['task_id'], 'kind': task['kind'],
                'task_fingerprint': gate.canonical(task), 'status': 'complete', 'counts': counts,
                'operation_seconds': samples, 'maxima': maxima, 'details': details,
                'actual_wall_seconds': 1., 'cache_reuse': False, 'outer_test_used': False,
                'official_adult_test_used': False, 'test_double_only': True}
            self.db.execute('INSERT INTO completed_tasks VALUES (?,?)', ('probe', task['task_id']))
            for kind, durations in samples.items():
                self.db.executemany('INSERT INTO operations(role,task,kind,status,seconds) VALUES (?,?,?,\'success\',?)',
                    [('probe', task['task_id'], kind, seconds) for seconds in durations])
        self.db.commit()

    def record(self, kind='selector_timing'):
        return next(self.records[t['task_id']] for t in self.tasks if t['kind'] == kind)

    def sha(self, path):
        # Stable fake file digest: this tests the gate contract after a caller
        # presents matching mock bytes/markers, not a cryptographic primitive.
        return hashlib.sha256(str(Path(path)).encode('utf8')).hexdigest()

    def read(self, path):
        path = Path(path)
        if path == Path(self.paths['timing_tasks_file']): return {'tasks': self.tasks}
        if path.name == 'timing.json' and path.parent.name in self.records: return self.records[path.parent.name]
        raise AssertionError('Unexpected JSON access in test: ' + str(path))

    def marker(self, path, *args, **kwargs):
        path = Path(path)
        if path.name != 'timing.json.sha256': raise AssertionError('Only mock marker reads are permitted')
        return 'wrong-digest' if str(path) in self.changed_markers else self.sha(path.with_suffix('')) + '\n'

    @contextmanager
    def mocked_io(self):
        expected_uri = 'file:' + Path(self.paths['ledger_file']).as_posix() + '?mode=ro'
        # Parameter name cannot be `uri` because sqlite's keyword has that name.
        def memory_connection(database, *args, **kwargs):
            if database != expected_uri or args or kwargs != {'uri': True}:
                raise AssertionError('Unexpected ledger access; only one mocked read-only URI is permitted')
            return self.db
        with ExitStack() as stack:
            stack.enter_context(patch.object(gate, 'validate_paths', return_value=True))
            stack.enter_context(patch.object(gate, 'prior_probe_usage', return_value={
                'counts':{'classifier':3855,'encoder':4036,'kmeans':240},'session_seconds':240.85351679999962,
                'operation_seconds_max':{'classifier':0,'encoder':.0005,'kmeans':0},
                'failed_sessions':1,'old_ledger_sha256':'preserved-prior-ledger'}))
            stack.enter_context(patch.object(gate, 'read_json', side_effect=self.read))
            stack.enter_context(patch.object(gate, 'sha256', side_effect=self.sha))
            stack.enter_context(patch.object(Path, 'is_file', lambda p: str(p) not in self.missing_files))
            stack.enter_context(patch.object(Path, 'read_text', lambda p,*args,**kwargs:self.marker(p,*args,**kwargs)))
            stack.enter_context(patch.object(Path, 'open', side_effect=AssertionError('No real file opens permitted')))
            stack.enter_context(patch.object(Path, 'write_text', side_effect=AssertionError('No file writes permitted')))
            stack.enter_context(patch.object(gate.sqlite3, 'connect', side_effect=memory_connection))
            yield

    def evidence(self):
        with self.mocked_io(): return gate.evidence(self.paths, 'test-double-freeze')


class GateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sentinels = [patch(target, forbidden_model_call) for target in (
            'sklearn.linear_model.LogisticRegression.fit', 'sklearn.linear_model.LogisticRegression.predict',
            'sklearn.linear_model.LogisticRegression.predict_proba', 'sklearn.cluster.KMeans.fit',
            'sklearn.cluster.KMeans.fit_predict')]
        for sentinel in cls.sentinels: sentinel.start()

    @classmethod
    def tearDownClass(cls):
        for sentinel in reversed(cls.sentinels): sentinel.stop()

    def setUp(self): self.fixture = SyntheticEvidence()
    def tearDown(self): self.fixture.db.close()

    def test_complete_synthetic_trace_reconciles_formula_and_counts(self):
        result = self.fixture.evidence()
        expected = 340.8535167999996 + 3.75 * (301350 * .01 + 1056 * .02 + 12 * .03 + 42840 * .02 + 12 * .03
            + 44132 * .001 + 152 * .002 + 384 * .004 + 96 * .005) + 3600
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['timing_tasks_complete'], 75)
        self.assertEqual(result['probe_operation_counts'], {'classifier': 7710, 'encoder': 8071, 'kmeans': 480})
        self.assertEqual(result['new_probe_operation_counts'], {'classifier': 3855, 'encoder': 4035, 'kmeans': 240})
        self.assertEqual(result['prior_failed_sessions_preserved'],1)
        self.assertEqual(result['measured_parameters']['T_probe_actual'],340.8535167999996)
        self.assertAlmostEqual(result['projected_total_seconds'], expected, places=9)
        self.assertEqual(len(result['source_sha256']), 150)
        self.assertFalse(result['execution_authorized'])

    def test_pure_summary_matches_independent_evidence_formula(self):
        from timing import summarize_timing
        config = json.loads((STAGE/'config.json').read_text('utf-8-sig'))
        records = list(self.fixture.records.values())
        summary = summarize_timing(records, config, probe_actual_seconds=100., old_probe_seconds=240.85351679999962)
        evidence = self.fixture.evidence()
        self.assertEqual(summary['predicted_seconds'], evidence['projected_total_seconds'])
        self.assertEqual(summary['new_probe_seconds'], 100.)
        self.assertEqual(summary['old_probe_seconds'], 240.85351679999962)
        self.assertFalse(summary['can_authorize_execution'])
        self.assertEqual(summarize_timing(records, config, probe_actual_seconds=100.)['status'], 'incomplete_ledger_wall_time_required')

    def test_timing_session_forwards_identity_and_events_without_granting_authority(self):
        from timing import TimingSession
        from unittest.mock import Mock
        parent = Mock(role='probe', task='fold0-full-reference-repeat0')
        parent.bind_classifier.return_value = {'bound': True}
        parent.record_event.return_value = 17
        wrapper = TimingSession(parent)
        self.assertEqual(wrapper.bind_classifier([1], [2], ['f'], {'phase':'test'}), {'bound':True})
        parent.bind_classifier.assert_called_once_with([1], [2], ['f'], {'phase':'test'})
        self.assertEqual(wrapper.record_event('complete_cycle', {'type':'reference'}, 1.),17)
        self.assertEqual(wrapper.counts, {'classifier':0,'encoder':0,'kmeans':0})

    def test_missing_or_duplicate_frozen_task_rejected(self):
        original = copy.deepcopy(self.fixture.tasks)
        for tasks in [original[:-1], original[:-1] + [copy.deepcopy(original[0])]]:
            self.fixture.tasks = tasks
            with self.subTest(count=len(tasks)), self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_missing_record_or_changed_marker_rejected(self):
        path = self.fixture.root / 'probe' / self.fixture.tasks[0]['task_id'] / 'timing.json'
        self.fixture.missing_files.add(str(path))
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()
        self.fixture.missing_files.clear()
        self.fixture.changed_markers.add(str(path.with_suffix('.json.sha256')))
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_wrong_raw_encoder_maximum_rejected(self):
        self.fixture.record()['maxima']['encoder'] = .0001
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_wrong_raw_utility_or_candidate_maximum_rejected(self):
        record = self.fixture.record()
        for key in ['utility_cycle', 'full_candidate_cycle']:
            original = record['maxima'][key]; record['maxima'][key] /= 2
            with self.subTest(key=key), self.assertRaises(ExecutionDenied): self.fixture.evidence()
            record['maxima'][key] = original

    def test_reference_outer_test_and_official_test_leakage_rejected(self):
        record = self.fixture.record('full_reference_timing')
        for key in ['outer_test_used', 'official_adult_test_used']:
            record[key] = True
            with self.subTest(key=key), self.assertRaises(ExecutionDenied): self.fixture.evidence()
            record[key] = False

    def test_reference_cannot_retain_scores_or_predictions(self):
        record = self.fixture.record('full_reference_timing')
        for key in ['utility_scores_retained', 'predictions_retained']:
            record['details'][key] = True
            with self.subTest(key=key), self.assertRaises(ExecutionDenied): self.fixture.evidence()
            record['details'][key] = False

    def test_overlap_declaration_must_match_task_role(self):
        for kind in ['selector_timing', 'full_reference_timing']:
            record = self.fixture.record(kind)
            original = record['details']['prediction_overlap_allowed']
            record['details']['prediction_overlap_allowed'] = not original
            with self.subTest(kind=kind), self.assertRaises(ExecutionDenied): self.fixture.evidence()
            record['details']['prediction_overlap_allowed'] = original

    def test_cache_reuse_rejected(self):
        self.fixture.record()['cache_reuse'] = True
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_nonfinite_maximum_rejected(self):
        self.fixture.record()['maxima']['game_setup'] = float('nan')
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_all_raw_operation_samples_must_be_finite_and_nonnegative(self):
        for kind in ['classifier', 'encoder', 'kmeans']:
            record = next(r for r in self.fixture.records.values() if r['counts'][kind] > 0)
            original = record['operation_seconds'][kind][0]
            for bad in [float('nan'), float('inf'), -1.]:
                record['operation_seconds'][kind][0] = bad
                with self.subTest(kind=kind,bad=bad), self.assertRaises(ExecutionDenied): self.fixture.evidence()
            record['operation_seconds'][kind][0] = original

    def test_wrong_counts_and_raw_count_rejected(self):
        record = self.fixture.record(); record['counts']['classifier'] = 63
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()
        record['counts']['classifier'] = 64; record['operation_seconds']['classifier'].pop()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_ledger_counts_and_completed_tasks_reconcile(self):
        self.fixture.db.execute('DELETE FROM operations WHERE id=1'); self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_missing_completed_probe_rejected(self):
        self.fixture.db.execute('DELETE FROM completed_tasks WHERE task=?', (self.fixture.tasks[0]['task_id'],)); self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_record_and_ledger_freeze_binding_rejected(self):
        record = self.fixture.record(); original = record['task_fingerprint']; record['task_fingerprint'] = 'other-task'
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()
        record['task_fingerprint'] = original
        self.fixture.db.execute("UPDATE metadata SET value='other-freeze'"); self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_probe_wall_time_must_cover_records_and_stay_within_two_hours(self):
        for seconds in [74., 7201.]:
            self.fixture.db.execute('UPDATE sessions SET seconds=?', (seconds,)); self.fixture.db.commit()
            with self.subTest(seconds=seconds), self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_old_probe_seconds_are_additional_to_new_two_hour_cap(self):
        self.fixture.db.execute('UPDATE sessions SET seconds=7200'); self.fixture.db.commit()
        result = self.fixture.evidence()
        self.assertEqual(result['new_probe_session_seconds'], 7200)
        self.assertAlmostEqual(result['measured_parameters']['T_probe_actual'], 7440.8535168)
        self.assertEqual(result['status'], 'passed')

    def test_failed_new_session_with_successful_operations_cannot_pass(self):
        self.fixture.db.execute("UPDATE sessions SET status='failed'"); self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_old_contract_gate_cannot_be_reused_for_exact_contract(self):
        published = self.fixture.evidence(); published.pop('resource_contract_id')
        with self.fixture.mocked_io(), self.assertRaises(ExecutionDenied):
            gate.verify_published_gate(self.fixture.paths, 'test-double-freeze', published)

    def test_encoder_attempt_4500_allowed_but_4501_denied(self):
        for number in range(465):
            self.fixture.db.execute("INSERT INTO operations(role,task,kind,status,seconds) VALUES ('probe',?,'encoder','success',.001)", (self.fixture.tasks[0]['task_id'],))
        self.fixture.db.commit()
        self.assertEqual(self.fixture.evidence()['new_probe_operation_counts']['encoder'], 4500)
        self.fixture.db.execute("INSERT INTO operations(role,task,kind,status,seconds) VALUES ('probe',?,'encoder','success',.001)", (self.fixture.tasks[0]['task_id'],))
        self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_failed_or_active_probe_rejected(self):
        self.fixture.db.execute("UPDATE operations SET status='failed' WHERE id=1"); self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()
        self.fixture.db.execute("UPDATE operations SET status='success' WHERE id=1")
        self.fixture.db.execute("UPDATE sessions SET status='active'"); self.fixture.db.commit()
        with self.assertRaises(ExecutionDenied): self.fixture.evidence()

    def test_later_main_failure_does_not_invalidate_passed_probe_gate(self):
        published = self.fixture.evidence()
        self.fixture.db.execute("INSERT INTO sessions VALUES (2,'main','failed',10)")
        self.fixture.db.execute("INSERT INTO operations(role,task,kind,status,seconds) VALUES ('main','fold-0','classifier','failed',10)")
        self.fixture.db.commit()
        with self.fixture.mocked_io():
            self.assertEqual(gate.verify_published_gate(self.fixture.paths, 'test-double-freeze', published), published)

    def test_published_gate_must_match_recomputed_evidence(self):
        published = self.fixture.evidence()
        for key,value in [('projected_total_seconds',1),('freeze_sha256','wrong'),('timing_tasks_complete',74),('source_sha256',{})]:
            changed = copy.deepcopy(published); changed[key] = value
            with self.subTest(key=key), self.fixture.mocked_io(), self.assertRaises(ExecutionDenied):
                gate.verify_published_gate(self.fixture.paths, 'test-double-freeze', changed)


if __name__ == '__main__': unittest.main(verbosity=2)
