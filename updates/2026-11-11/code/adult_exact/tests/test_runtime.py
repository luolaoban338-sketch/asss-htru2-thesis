"""No-fit controls regression checks. Only in-memory SQLite is permitted here."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import copy
import contextlib
import io
import json
import math
import sqlite3
import sys
import unittest
from unittest.mock import patch, Mock

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
import runtime

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
CONFIG = json.loads((STAGE / 'config.json').read_text(encoding='utf-8-sig'))


def forbidden_fit(*args, **kwargs):
    raise AssertionError('REAL FIT/PREDICTION IS FORBIDDEN IN NO-FIT TESTS')


def auth(role='main'):
    caps = CONFIG['budget']['future_timing_probe']['hard_caps'] if role == 'probe' else CONFIG['budget']['execution_role_caps'][role]
    return {'explicit_execution_authorized': True, 'role': role, 'task': 'fold-0',
            'freeze_sha256': 'freeze', 'config_sha256': 'digest', 'paths_sha256': 'digest',
            'authorization_id': 'test-only-in-memory', 'user_instruction': 'MOCK ONLY NO EXECUTION',
            'issued_at': (NOW - timedelta(minutes=1)).isoformat(),
            'expires_at': (NOW + timedelta(minutes=30)).isoformat(),
            'entry_used_percent': 10, 'stop_used_percent': 14, 'ceiling_used_percent': 16,
            'attempt_caps': copy.deepcopy(caps), 'quota_observation_file': 'never-opened.json'}


def memory_session(role='main'):
    session = runtime.GuardedExecution(role, 'fold-0', None)
    session.db = sqlite3.connect(':memory:')
    session.db.executescript('''
CREATE TABLE operations(id INTEGER PRIMARY KEY,session_id INTEGER,role TEXT,task TEXT,kind TEXT,phase TEXT,metadata TEXT,status TEXT,seconds REAL,error TEXT);
CREATE TABLE counters(role TEXT,kind TEXT,n INTEGER,PRIMARY KEY(role,kind));
CREATE TABLE completed_tasks(role TEXT,task TEXT,PRIMARY KEY(role,task));
CREATE TABLE events(id INTEGER PRIMARY KEY,session_id INTEGER,name TEXT,metadata TEXT,seconds REAL);
CREATE TABLE type_counters(role TEXT,bucket TEXT,n INTEGER,PRIMARY KEY(role,bucket));
CREATE TABLE completion_evidence(role TEXT,task TEXT,path TEXT,sha256 TEXT,PRIMARY KEY(role,task));
''')
    session.config = copy.deepcopy(CONFIG)
    session.auth = auth(role)
    session.session_id = 1
    session.active = True
    session.initial_task_counts = {k: 0 for k in runtime.KINDS}
    session.checkpoint = lambda: None
    return session

def synthetic_bound(session):
    metadata={'phase':'mock','role':session.role,'task':session.task,'classifier_bucket':'pool'}
    metadata['binding_token']=session._binding_digest(metadata)
    return metadata


class NoFitSentinels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # If a future refactor accidentally reaches model operations, the suite
        # fails before a real fit/predict. This suite never opens a disk ledger.
        cls.sentinels = [
            patch('sklearn.linear_model.LogisticRegression.fit', forbidden_fit),
            patch('sklearn.linear_model.LogisticRegression.predict_proba', forbidden_fit),
            patch('sklearn.linear_model.LogisticRegression.predict', forbidden_fit),
            patch('sklearn.cluster.KMeans.fit', forbidden_fit),
            patch('sklearn.cluster.KMeans.fit_predict', forbidden_fit),
        ]
        for item in cls.sentinels: item.start()

    @classmethod
    def tearDownClass(cls):
        for item in reversed(cls.sentinels): item.stop()


class AuthorizationTests(NoFitSentinels):
    def test_default_entry_rejects_before_opening_any_ledger(self):
        with patch.object(runtime.sqlite3, 'connect', side_effect=AssertionError('No disk ledger allowed')):
            with self.assertRaises(runtime.ExecutionDenied):
                runtime.preflight('main', 'fold-0', None)

    def test_valid_mock_authorization(self):
        with patch.object(runtime, 'sha256', return_value='digest'):
            self.assertTrue(runtime.validate_authorization(auth(), CONFIG, {}, 'main', 'fold-0', 'freeze', now=NOW))

    def test_numeric_one_is_not_explicit_boolean_authorization(self):
        value = auth(); value['explicit_execution_authorized'] = 1
        with patch.object(runtime, 'sha256', return_value='digest'):
            with self.assertRaises(runtime.ExecutionDenied):
                runtime.validate_authorization(value, CONFIG, {}, 'main', 'fold-0', 'freeze', now=NOW)

    def test_expired_authorization(self):
        value = auth(); value['expires_at'] = NOW.isoformat()
        with patch.object(runtime, 'sha256', return_value='digest'):
            with self.assertRaises(runtime.ExecutionDenied):
                runtime.validate_authorization(value, CONFIG, {}, 'main', 'fold-0', 'freeze', now=NOW)

    def test_wrong_role_or_freeze(self):
        for key, changed in [('role', 'audit'), ('freeze_sha256', 'other')]:
            value = auth(); value[key] = changed
            with self.subTest(key=key), patch.object(runtime, 'sha256', return_value='digest'):
                with self.assertRaises(runtime.ExecutionDenied):
                    runtime.validate_authorization(value, CONFIG, {}, 'main', 'fold-0', 'freeze', now=NOW)

    def test_role_attempt_limit_not_expandable(self):
        value = auth(); value['attempt_caps']['classifier'] += 1
        with patch.object(runtime, 'sha256', return_value='digest'):
            with self.assertRaises(runtime.ExecutionDenied):
                runtime.validate_authorization(value, CONFIG, {}, 'main', 'fold-0', 'freeze', now=NOW)

    def test_nonfinite_account_envelope(self):
        for bad in [float('nan'), float('inf'), True, -1]:
            value = auth(); value['entry_used_percent'] = bad
            with self.subTest(value=bad), patch.object(runtime, 'sha256', return_value='digest'):
                with self.assertRaises(runtime.ExecutionDenied):
                    runtime.validate_authorization(value, CONFIG, {}, 'main', 'fold-0', 'freeze', now=NOW)

    def test_quota_rejects_new_period_or_regression_below_entry(self):
        observation = {'authorization_id': auth()['authorization_id'], 'observed_at': NOW.isoformat(), 'used_percent': 1}
        with patch.object(runtime, 'read_json', return_value=observation):
            with self.assertRaises(runtime.ExecutionDenied):
                runtime.validate_quota(auth(), now=NOW)

    def test_quota_stop_stale_and_wrong_identity(self):
        cases = [
            {'used_percent':14},
            {'observed_at':(NOW-timedelta(seconds=301)).isoformat()},
            {'authorization_id':'another'},
            {'used_percent':float('nan')},
        ]
        for updates in cases:
            value={'authorization_id':auth()['authorization_id'],'observed_at':NOW.isoformat(),'used_percent':11}
            value.update(updates)
            with self.subTest(updates=updates), patch.object(runtime,'read_json',return_value=value):
                with self.assertRaises(runtime.ExecutionDenied):runtime.validate_quota(auth(),now=NOW)

    def test_out_of_schedule_task_denied_even_with_authorization(self):
        import freeze
        for role,task in [('main','fold-9'),('replay','fold-1'),('audit','unknown-bundle')]:
            value=auth(role);value['task']=task
            paths={'stop_file':str(STAGE/'nonexistent-mock-stop')}
            def reader(path):
                if Path(path).name=='config.json':return CONFIG
                if Path(path).name=='paths.json':return paths
                return value
            with self.subTest(role=role,task=task),patch.object(runtime,'read_json',side_effect=reader),patch.object(runtime,'prior_probe_usage',return_value={}),patch.object(Path,'exists',return_value=False),patch.object(freeze,'verify_freeze',return_value={'freeze_sha256':'freeze'}),patch.object(runtime,'validate_paths',return_value=True),patch.object(runtime,'validate_authorization',return_value=True),patch.object(runtime,'validate_quota',return_value=11),patch.object(runtime,'validate_gate',return_value={}):
                with self.assertRaisesRegex(runtime.ExecutionDenied,'frozen schedule'):
                    runtime.preflight(role,task,'mock-auth-never-read.json')


class PathTests(NoFitSentinels):
    def test_relative_escape_rejected(self):
        session=runtime.GuardedExecution('main','fold-0',None)
        session.paths={'run_root':str(STAGE/'mock-run-no-write')}
        with self.assertRaises(runtime.ExecutionDenied):session.artifact_path('../escape.json')

    def test_task_junction_cannot_redirect_storage(self):
        session=runtime.GuardedExecution('main','fold-0',None)
        run_root=Path('F:/ChatGPT/论文大数据/runs/adult_reduced_20261107')
        session.paths={'run_root':str(run_root)}
        base=run_root/'main'/'fold-0'
        escaped=STAGE/'simulated-junction-target-no-write'
        original_resolve=Path.resolve
        def mocked_resolve(path,*args,**kwargs):
            try: suffix=path.relative_to(base)
            except ValueError:return original_resolve(path,*args,**kwargs)
            return original_resolve(escaped/suffix,*args,**kwargs)
        with patch.object(Path,'resolve',mocked_resolve):
            with self.assertRaises(runtime.ExecutionDenied):session.artifact_path('model.json')


class GateTests(NoFitSentinels):
    def gate(self):
        return {'status':'passed','freeze_sha256':'freeze','projected_total_seconds':100,
                'timing_tasks_complete':75,'independent_checks_passed':True,'source_sha256':{'example.json':'digest'},
                'measured_parameters':{'t_utility':.1,'t_full_candidate':.1,'t_reference':.1,
                  't_encoder':.01,'t_kmeans':.01,'t_game_setup':.01,'t_selector_setup':.01,'T_probe_actual':1}}

    def test_nonfinite_or_missing_measured_parameters_rejected(self):
        for bad in [float('nan'),float('inf'),-1,None]:
            gate=self.gate()
            if bad is None:del gate['measured_parameters']['t_utility']
            else:gate['measured_parameters']['t_utility']=bad
            paths={'runtime_gate_file':str(STAGE/'mock-gate-never-written.json'),'run_root':str(STAGE)}
            with self.subTest(value=bad),patch.object(Path,'is_file',return_value=True),patch.object(runtime,'read_json',return_value=gate),patch.object(runtime,'sha256',return_value='digest'):
                with self.assertRaises(runtime.ExecutionDenied):runtime.validate_gate(paths,'freeze')


class BudgetTests(NoFitSentinels):
    def test_failed_operation_still_consumes_budget(self):
        s=memory_session()
        try:
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                with s.operation('classifier',synthetic_bound(s)):raise RuntimeError('synthetic failure')
            self.assertEqual(s._count('main','classifier'),1)
            self.assertEqual(s.db.execute('SELECT status FROM operations').fetchone()[0],'failed')
        finally:s.db.close()

    def test_main_cannot_consume_reserved_role_capacity(self):
        s=memory_session()
        try:
            cap=s.config['budget']['execution_role_caps']['main']['classifier']
            s.db.execute('INSERT INTO counters VALUES (?,?,?)',('main','classifier',cap));s.db.commit()
            with self.assertRaises(runtime.BudgetStop):
                with s.operation('classifier',synthetic_bound(s)):self.fail('Operation must not start')
            self.assertEqual(s._count('replay','classifier'),0)
            self.assertEqual(s._count('audit','classifier'),0)
        finally:s.db.close()

    def test_finite_invocation_attempt_caps(self):
        s=memory_session()
        try:
            s.auth['attempt_caps']['encoder']=1
            with s.operation('encoder',{'phase':'mock'}):pass
            with self.assertRaises(runtime.BudgetStop):
                with s.operation('encoder',{'phase':'mock'}):self.fail('Operation must not start')
            self.assertEqual(s._count_task('encoder'),1)
        finally:s.db.close()

    def test_role_and_task_metadata_mismatch_rejected(self):
        s=memory_session()
        try:
            for metadata in [{'phase':'mock','role':'audit'},{'phase':'mock','task':'fold-4'}]:
                with self.subTest(metadata=metadata),self.assertRaises(runtime.ExecutionDenied):
                    with s.operation('encoder',metadata):self.fail('Operation must not start')
            self.assertEqual(s.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0],0)
        finally:s.db.close()

    def test_partial_task_gets_no_projection_credit(self):
        s=memory_session()
        try:
            s.gate={'measured_parameters':{'t_utility':1,'t_full_candidate':1,'t_reference':1,'t_encoder':1,'t_kmeans':1,'t_game_setup':1,'t_selector_setup':1}}
            before=s.remaining_projection_seconds()
            with s.operation('encoder',{'phase':'mock'}):pass
            self.assertEqual(s.remaining_projection_seconds(),before)
            s.db.execute('INSERT INTO completed_tasks VALUES (?,?)',('main','fold-0'));s.db.commit()
            self.assertLess(s.remaining_projection_seconds(),before)
        finally:s.db.close()

    def test_slower_audit_cycle_increases_remaining_projection(self):
        for phase in ['audit_final','audit_reference']:
            s=memory_session()
            try:
                s.gate={'measured_parameters':{'t_utility':1,'t_full_candidate':1,'t_reference':1,'t_encoder':1,'t_kmeans':1,'t_game_setup':1,'t_selector_setup':1}}
                before=s.remaining_projection_seconds()
                s.record_event('audit_complete_cycle',{'phase':phase},2)
                with self.subTest(phase=phase):self.assertGreater(s.remaining_projection_seconds(),before)
            finally:s.db.close()

    def test_faster_audit_cycle_never_reduces_projection(self):
        s=memory_session()
        try:
            s.gate={'measured_parameters':{'t_utility':1,'t_full_candidate':1,'t_reference':1,'t_encoder':1,'t_kmeans':1,'t_game_setup':1,'t_selector_setup':1}}
            before=s.remaining_projection_seconds()
            s.record_event('audit_complete_cycle',{'phase':'audit_reference'},.1)
            s.record_event('audit_complete_cycle',{'phase':'audit_final'},.1)
            self.assertEqual(s.remaining_projection_seconds(),before)
        finally:s.db.close()

    def test_invalid_audit_duration_rejected(self):
        s=memory_session()
        try:
            s.gate={'measured_parameters':{'t_utility':1,'t_full_candidate':1,'t_reference':1,'t_encoder':1,'t_kmeans':1,'t_game_setup':1,'t_selector_setup':1}}
            for duration in [float('inf'),-1]:
                s.db.execute('DELETE FROM events')
                s.db.execute('INSERT INTO events VALUES (1,1,?,?,?)',('audit_complete_cycle',json.dumps({'phase':'audit_final'}),duration));s.db.commit()
                with self.subTest(duration=duration),self.assertRaises(runtime.ExecutionDenied):s.refresh_observed_rates()
        finally:s.db.close()


class CheckpointTests(NoFitSentinels):
    def session(self,role='main'):
        s=runtime.GuardedExecution(role,'fold-0',None)
        s.active=True;s.started=0;s.prior_seconds=0;s.prior_probe_seconds=0
        s.paths={'stop_file':str(STAGE/'mock-stop-never-created')}
        s.auth=auth(role)
        return s

    def checkpoint(self,s,elapsed):
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return NOW
        with patch.object(Path,'exists',return_value=False),patch.object(runtime,'datetime',Clock),patch.object(runtime,'validate_quota',return_value=11),patch.object(runtime.time,'perf_counter',return_value=elapsed):
            return s.checkpoint()

    def test_inactive_context_denied(self):
        s=self.session();s.active=False
        with self.assertRaises(runtime.ExecutionDenied):self.checkpoint(s,0)

    def test_stop_file_denies(self):
        s=self.session()
        with patch.object(Path,'exists',return_value=True):
            with self.assertRaisesRegex(runtime.BudgetStop,'STOP_REQUESTED'):s.checkpoint()

    def test_cumulative_wall_cap_denies(self):
        s=self.session();s.prior_seconds=172799
        with self.assertRaisesRegex(runtime.BudgetStop,'48-hour'):self.checkpoint(s,1)

    def test_finite_main_invocation_denies(self):
        s=self.session()
        with self.assertRaisesRegex(runtime.BudgetStop,'invocation'):self.checkpoint(s,3600)

    def test_probe_cumulative_cap_denies(self):
        s=self.session('probe');s.prior_probe_seconds=7199
        with self.assertRaisesRegex(runtime.BudgetStop,'2-hour'):self.checkpoint(s,1)

    def test_probe_invocation_cap_denies(self):
        s=self.session('probe')
        with self.assertRaisesRegex(runtime.BudgetStop,'invocation'):self.checkpoint(s,1800)

    def test_normal_mock_checkpoint_passes(self):
        self.checkpoint(self.session(),10)


class EntryPointTests(NoFitSentinels):
    def test_default_status_never_dispatches_execution(self):
        import run
        with patch.object(run,'preflight',side_effect=AssertionError('No execution preflight expected')),patch.object(run,'worker',side_effect=AssertionError('No worker expected')),patch.object(run,'launch',side_effect=AssertionError('No launcher expected')),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(run.main([]),0)
        self.assertEqual(json.loads(stream.getvalue())['status'],'no_execution_requested')

    def test_execute_flag_alone_cannot_authorize(self):
        import run
        with patch.object(run,'launch',side_effect=AssertionError('Launcher must not start')):
            with self.assertRaises(runtime.ExecutionDenied):run.main(['execute','--execute-authorized'])

    def test_authorization_path_without_explicit_flag_denied(self):
        import run
        with patch.object(run,'launch',side_effect=AssertionError('Launcher must not start')):
            with self.assertRaises(runtime.ExecutionDenied):
                run.main(['execute','--role','main','--task','fold-0','--authorization','never-read.json'])

    def test_internal_worker_requires_matching_launcher_token(self):
        import run
        args=['_worker','--role','main','--task','fold-0','--authorization','never-read.json','--execute-authorized']
        with patch.object(run,'worker',side_effect=AssertionError('Worker must not start')):
            for environment,supplied in [({},None),({'ADULT_WORKER_TOKEN':'expected'},'wrong')]:
                with self.subTest(environment=environment),patch.dict(run.os.environ,environment,clear=True):
                    with self.assertRaises(runtime.ExecutionDenied):
                        run.main(args+(['--watchdog-token',supplied] if supplied else []))

    def test_watchdog_enforces_authorization_expiry_without_waiting_for_fit(self):
        import run
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return NOW+timedelta(hours=1)
        paths={'ledger_file':str(STAGE/'never-created.sqlite3'),'run_root':str(STAGE/'mock-watchdog-no-write'),
               'stop_file':str(STAGE/'never-created-stop')}
        process=Mock(pid=123456,returncode=-9)
        process.poll.side_effect=[None,None]
        with patch.object(run,'preflight',return_value=(CONFIG,paths,auth(),{'freeze_sha256':'freeze'},{})),patch.object(run,'prior_probe_usage',return_value={'session_seconds':1.1159227999999075}),patch.object(Path,'exists',return_value=False),patch.object(Path,'mkdir'),patch.object(Path,'open',side_effect=lambda *a,**kw:io.StringIO()),patch.object(Path,'write_text') as receipt_write,patch.object(run.subprocess,'Popen',return_value=process),patch.object(run.subprocess,'run') as taskkill,patch.object(run,'datetime',Clock),patch.object(run,'validate_quota',side_effect=AssertionError('Expiry must be detected first')),patch.object(run.sqlite3,'connect',side_effect=AssertionError('No disk ledger allowed')):
            result=run.launch('main','fold-0','never-read-authorization.json')
        self.assertTrue(result['forced_termination'])
        self.assertTrue(result['recovery_audit_required'])
        self.assertIn('authorization expiry',result['reason'])
        taskkill.assert_called_once()
        process.kill.assert_called_once()
        receipt_write.assert_called_once()


class PriorUsageTests(NoFitSentinels):
    def test_preserved_usage_reconciles_without_sqlite(self):
        with patch.object(runtime.sqlite3,'connect',side_effect=AssertionError('No historical SQLite connection')):
            prior=runtime.prior_probe_usage()
        self.assertEqual(prior['counts'],{'classifier':3855,'encoder':4036,'kmeans':240})
        self.assertAlmostEqual(prior['session_seconds'],240.85351679999962)
        self.assertEqual(prior['failed_sessions'],1)
        self.assertEqual(prior['interrupted_sessions'],1)
        self.assertFalse(prior['exact_interrupted_duration_known'])
        self.assertEqual(CONFIG['budget']['future_timing_probe']['hard_caps']['encoder'],4500)
        self.assertEqual(CONFIG['budget']['total_future_caps_including_probe']['encoder'],351770)
    def test_prior_ledger_hash_change_denied_before_sqlite(self):
        with patch.object(runtime,'sha256',return_value='changed'),patch.object(runtime.sqlite3,'connect',side_effect=AssertionError('No historical access')):
            with self.assertRaises(runtime.ExecutionDenied):runtime.prior_probe_usage()
    def test_prior_count_tampering_denied(self):
        original=runtime.read_json
        def altered(path):
            value=original(path)
            if Path(path).name=='prior_probe_usage.json':value['counts']['encoder']=0
            return value
        with patch.object(runtime,'read_json',side_effect=altered):
            with self.assertRaises(runtime.ExecutionDenied):runtime.prior_probe_usage()

if __name__=='__main__':unittest.main()
