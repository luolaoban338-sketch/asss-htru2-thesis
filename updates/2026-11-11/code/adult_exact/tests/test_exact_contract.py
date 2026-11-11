"""Independent exact-resource adversarial tests; only :memory: SQL, never fits."""
from pathlib import Path
import copy
import json
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
import runtime
from resource_binding import MembershipBinding


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


class ExactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read(STAGE / 'config.json')
        cls.paths = read(STAGE / 'paths.json')
        cls.parts = read(cls.paths['partitions_file'])
        games = read(cls.paths['games_file'])
        for key, context in cls.parts['contexts'].items():
            context['games'] = games['contexts'][key]['games']
        cls.tasks = read(cls.paths['timing_tasks_file'])['tasks']
        schema = read(cls.config['data_source']['dictionary'])
        cls.features = tuple(r['field'] for r in schema if r['role'] == 'predictor')
        cls.binding = MembershipBinding(cls.parts, cls.tasks, cls.features)
        cls.outer = cls.parts['outer_folds'][0]

    def session(self, role='main', task='fold-0'):
        s = runtime.GuardedExecution(role, task, None)
        s.db = sqlite3.connect(':memory:')
        s.db.executescript('''
CREATE TABLE operations(id INTEGER PRIMARY KEY,session_id INTEGER,role TEXT,task TEXT,kind TEXT,phase TEXT,metadata TEXT,status TEXT,seconds REAL,error TEXT);
CREATE TABLE counters(role TEXT,kind TEXT,n INTEGER,PRIMARY KEY(role,kind));
CREATE TABLE type_counters(role TEXT,bucket TEXT,n INTEGER,PRIMARY KEY(role,bucket));
CREATE TABLE completed_tasks(role TEXT,task TEXT,PRIMARY KEY(role,task));
CREATE TABLE completion_evidence(role TEXT,task TEXT,path TEXT,sha256 TEXT,PRIMARY KEY(role,task));
CREATE TABLE events(id INTEGER PRIMARY KEY,session_id INTEGER,name TEXT,metadata TEXT,seconds REAL);
''')
        self.addCleanup(s.db.close)
        s.config = copy.deepcopy(self.config)
        s.paths = self.paths
        s.auth = {'attempt_caps': {'classifier':1000000, 'encoder':1000000, 'kmeans':1000000}}
        s.session_id = 1
        s.active = True
        s.gate = None
        s.initial_task_counts = {k:0 for k in runtime.KINDS}
        s.membership = self.binding
        s.checkpoint = lambda:None
        return s

    def ref(self, s, **updates):
        meta = {'phase':'audit_reference' if s.role == 'audit' else 'reference', 'fold':0}
        meta.update(updates)
        return s.bind_classifier(self.outer['original_train']['positions'], self.outer['test']['positions'], self.features, meta)

    def pool(self, s, **updates):
        meta = {'phase':'audit_final' if s.role == 'audit' else 'final', 'fold':0}
        meta.update(updates)
        return s.bind_classifier(self.outer['pool']['positions'], self.outer['test']['positions'], self.features, meta)

    def utility(self, s, **updates):
        cid = self.outer['inner_context_ids'][0]
        game = self.parts['contexts'][cid]['games'][0]
        fold = game['utility_folds'][0]
        mask = game['nonempty_coalition_masks']['full'][0]
        features = tuple(f for i,f in enumerate(self.features) if mask & (1 << i))
        meta = {'phase':'audit_utility' if s.role=='audit' else 'utility', 'fold':0,
                'context':cid,'resample':0,'method':'full','utility_fold':0,'mask':mask}
        meta.update(updates)
        return s.bind_classifier(fold['train_positions'], fold['validation_positions'], features, meta)

    def count(self, s, role, bucket):
        value = s.db.execute('SELECT n FROM type_counters WHERE role=? AND bucket=?',(role,bucket)).fetchone()
        return value[0] if value else 0

    def test_actual_membership_assigns_reference_and_pool_across_roles(self):
        for role,task in [('main','fold-0'),('replay','fold-0'),('audit','main-fold-0'),('audit','replay-fold-0')]:
            with self.subTest(role=role,task=task):
                s = self.session(role,task)
                self.assertEqual(self.ref(s)['classifier_bucket'],'reference')
                self.assertEqual(self.pool(s)['classifier_bucket'],'pool')
                self.assertEqual(self.utility(s)['classifier_bucket'],'pool')

    def test_reference_cannot_be_mislabeled_as_pool(self):
        s=self.session()
        for updates in ({'phase':'final'},{'classifier_bucket':'pool'}):
            with self.subTest(updates=updates), self.assertRaises(runtime.ExecutionDenied):self.ref(s,**updates)

    def test_pool_cannot_be_mislabeled_as_reference(self):
        s=self.session()
        for updates in ({'phase':'reference'},{'classifier_bucket':'reference'}):
            with self.subTest(updates=updates), self.assertRaises(runtime.ExecutionDenied):self.pool(s,**updates)

    def test_role_task_and_fold_mismatch_are_rejected(self):
        s=self.session()
        for updates in ({'fold':1},{'role':'audit'},{'task':'fold-1'}):
            with self.subTest(updates=updates), self.assertRaises(runtime.ExecutionDenied):self.pool(s,**updates)

    def test_invalid_replay_and_audit_task_identities_are_rejected(self):
        for role,task in [('replay','fold-1'),('audit','fold-0'),('main','fold-9')]:
            with self.subTest(role=role,task=task),self.assertRaises(runtime.ExecutionDenied):self.pool(self.session(role,task))

    def test_duplicate_and_outside_members_are_rejected(self):
        s=self.session();pool=self.outer['pool']['positions'];test=self.outer['test']['positions']
        for rows in (pool+[pool[0]], pool+[test[0]], [], self.outer['original_train']['positions']):
            with self.subTest(length=len(rows)),self.assertRaises(runtime.ExecutionDenied):
                s.bind_classifier(rows,test,self.features,{'phase':'final','fold':0})

    def test_complete_reference_requires_all_features_and_complete_test(self):
        s=self.session();train=self.outer['original_train']['positions'];test=self.outer['test']['positions']
        for rows,evaluation,features in [(train[:-1],test,self.features),(train,test[:-1],self.features),(train,test,self.features[:-1])]:
            with self.subTest(train=len(rows),test=len(evaluation),features=len(features)),self.assertRaises(runtime.ExecutionDenied):
                s.bind_classifier(rows,evaluation,features,{'phase':'reference','fold':0})

    def test_undeclared_utility_coalition_is_rejected(self):
        s=self.session();cid=self.outer['inner_context_ids'][0];game=self.parts['contexts'][cid]['games'][0]
        absent=next(m for m in range(1,4096) if m not in game['nonempty_coalition_masks']['full'])
        features=tuple(f for i,f in enumerate(self.features) if absent&(1<<i))
        uf=game['utility_folds'][0]
        with self.assertRaises(runtime.ExecutionDenied):
            s.bind_classifier(uf['train_positions'],uf['validation_positions'],features,
                {'phase':'utility','fold':0,'context':cid,'resample':0,'method':'full','utility_fold':0,'mask':absent})

    def test_utility_context_resample_fold_mask_and_method_are_bound(self):
        s=self.session()
        changes=[{'context':'adult_reduced:outer1:inner0'},{'resample':99},{'utility_fold':9},{'mask':4096},{'method':'unfrozen'}]
        for updates in changes:
            with self.subTest(updates=updates),self.assertRaises(runtime.ExecutionDenied):self.utility(s,**updates)

    def test_candidate_context_is_bound_to_real_inner_membership(self):
        s=self.session();cid=self.outer['inner_context_ids'][0];context=self.parts['contexts'][cid]
        good={'phase':'candidate','fold':0,'inner':0,'method':'full'}
        self.assertEqual(s.bind_classifier(context['train'],context['validation'],self.features,good)['classifier_bucket'],'pool')
        with self.assertRaises(runtime.ExecutionDenied):
            s.bind_classifier(context['train'],context['validation'],self.features,{**good,'inner':1})

    def test_all_probe_task_types_bind_without_using_outer_test(self):
        for spec in self.tasks:
            s=self.session('probe',spec['task_id'])
            phase='reference_timing' if spec['kind']=='full_reference_timing' else 'candidate'
            if phase=='reference_timing' or spec['method']=='full':
                meta=s.bind_classifier(spec['train_positions'],spec['prediction_positions'],self.features,{'phase':phase,'fold':spec['fold']})
                self.assertEqual(meta['classifier_bucket'],'reference' if phase=='reference_timing' else 'pool')

    def test_binding_cannot_be_injected_or_modified(self):
        s=self.session()
        with self.assertRaises(runtime.ExecutionDenied):self.pool(s,binding_token='forged')
        original=self.pool(s)
        for key,value in [('classifier_bucket','reference'),('training_rows',1),('phase','reference'),('fold',1)]:
            corrupted={**original,key:value}
            with self.subTest(key=key),self.assertRaises(runtime.ExecutionDenied):
                with s.operation('classifier',corrupted):self.fail('forged binding reached operation')
        self.assertEqual(s._count('main','classifier'),0)

    def test_another_session_cannot_reuse_a_binding(self):
        first=self.session();second=self.session()
        with self.assertRaises(runtime.ExecutionDenied):
            with second.operation('classifier',self.pool(first)):self.fail('cross-session binding accepted')

    def test_unbound_classifier_is_rejected_before_accounting(self):
        s=self.session()
        with self.assertRaises(runtime.ExecutionDenied):
            with s.operation('classifier',{'phase':'final','fold':0,'classifier_bucket':'pool'}):self.fail('unbound operation')
        self.assertEqual(s._count('main','classifier'),0)

    def test_reference_and_pool_buckets_cannot_borrow(self):
        for bucket in ('reference','pool'):
            s=self.session();cap=s.config['budget']['classifier_buckets_by_role']['main'][bucket]['total_cap']
            s.db.execute('INSERT INTO type_counters VALUES (?,?,?)',('main',bucket,cap));s.db.commit()
            meta=self.ref(s) if bucket=='reference' else self.pool(s)
            with self.subTest(bucket=bucket),self.assertRaises(runtime.BudgetStop):
                with s.operation('classifier',meta):self.fail('type cap exceeded')
            other='pool' if bucket=='reference' else 'reference'
            self.assertEqual(self.count(s,'main',other),0)
            self.assertEqual(s._count('main','classifier'),0)

    def test_role_cap_prevents_using_replay_or_audit_reserves(self):
        s=self.session();cap=s.config['budget']['execution_role_caps']['main']['classifier']
        s.db.execute('INSERT INTO counters VALUES (?,?,?)',('main','classifier',cap));s.db.commit()
        with self.assertRaises(runtime.BudgetStop):
            with s.operation('classifier',self.pool(s)):self.fail('main used other role budget')
        self.assertEqual(self.count(s,'main','pool'),0)
        self.assertEqual(s._count('replay','classifier'),0)
        self.assertEqual(s._count('audit','classifier'),0)

    def test_exact_last_slot_succeeds_and_next_attempt_stops(self):
        s=self.session();cap=s.config['budget']['classifier_buckets_by_role']['main']['reference']['total_cap']
        s.db.execute('INSERT INTO type_counters VALUES (?,?,?)',('main','reference',cap-1));s.db.commit()
        with s.operation('classifier',self.ref(s)):pass
        self.assertEqual(self.count(s,'main','reference'),cap)
        with self.assertRaises(runtime.BudgetStop):
            with s.operation('classifier',self.ref(s)):self.fail('one past cap')

    def test_failed_and_interrupted_attempts_keep_both_counts(self):
        for exc in (RuntimeError('synthetic failed fit'),KeyboardInterrupt('synthetic interruption')):
            s=self.session()
            with self.subTest(exception=type(exc).__name__),self.assertRaises(type(exc)):
                with s.operation('classifier',self.ref(s)):
                    self.assertEqual(s._count('main','classifier'),1)
                    self.assertEqual(self.count(s,'main','reference'),1)
                    self.assertEqual(s.db.execute('SELECT status FROM operations').fetchone()[0],'started')
                    raise exc
            self.assertEqual(s.db.execute('SELECT status FROM operations').fetchone()[0],'failed')
            self.assertEqual(s._count('main','classifier'),1)
            self.assertEqual(self.count(s,'main','reference'),1)

    def test_type_reservation_rolls_back_when_operation_insert_fails(self):
        s=self.session()
        s.db.executescript("CREATE TRIGGER reject_synthetic_operation BEFORE INSERT ON operations BEGIN SELECT RAISE(ABORT, 'synthetic insert failure'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            with s.operation('classifier',self.ref(s)):self.fail('missing synthetic schema')
        self.assertEqual(self.count(s,'main','reference'),0)
        self.assertEqual(s._count('main','classifier'),0)

    def test_missing_committed_evidence_is_rejected(self):
        s=self.session();s.db.execute('INSERT INTO completed_tasks VALUES (?,?)',('main','fold-0'));s.db.commit()
        with self.assertRaises(runtime.ExecutionDenied):s.verify_completion_evidence()

    def test_status_only_fake_bundle_cannot_earn_projection_credit(self):
        s=self.session()
        with patch.object(runtime,'read_json',return_value={'status':'complete_pending_independent_audit'}),patch.object(runtime,'sha256',return_value='digest'),patch.object(Path,'read_text',return_value='digest'):
            with self.assertRaises(runtime.ExecutionDenied):s.mark_completed()
        self.assertEqual(s.db.execute('SELECT COUNT(*) FROM completed_tasks').fetchone()[0],0)

    def test_wrong_fold_or_role_bundle_cannot_earn_projection_credit(self):
        for updates in ({'fold':1},{'role':'replay'}):
            s=self.session()
            value={'status':'complete_pending_independent_audit','fold':0,'role':'main','artifact_sha256':{'decisions.json':'digest'},**updates}
            with self.subTest(updates=updates),patch.object(runtime,'read_json',return_value=value),patch.object(runtime,'sha256',return_value='digest'),patch.object(Path,'read_text',return_value='digest'):
                with self.assertRaises(runtime.ExecutionDenied):s.mark_completed()
            self.assertEqual(s.db.execute('SELECT COUNT(*) FROM completed_tasks').fetchone()[0],0)

    def test_wrong_hash_marker_cannot_commit_completion(self):
        s=self.session()
        value={'status':'complete_pending_independent_audit','fold':0,'role':'main','artifact_sha256':{'decisions.json':'digest'}}
        with patch.object(runtime,'read_json',return_value=value),patch.object(runtime,'sha256',return_value='digest'),patch.object(Path,'read_text',return_value='different'):
            with self.assertRaises(runtime.ExecutionDenied):s.mark_completed()
        self.assertEqual(s.db.execute('SELECT COUNT(*) FROM completion_evidence').fetchone()[0],0)

    def test_pool_and_reference_audit_rates_update_distinct_maxima(self):
        s=self.session('audit','main-fold-0')
        s.record_event('audit_complete_cycle',{'phase':'audit_reference'},2.)
        s.record_event('audit_complete_cycle',{'phase':'audit_final'},1.)
        s.record_event('audit_complete_cycle',{'phase':'audit_reference'},.5)
        self.assertEqual(s.observed_rates['t_direct_audit_reference'],2.)
        self.assertEqual(s.observed_rates['t_direct_audit_cycle'],1.)

    def test_bad_resource_rates_are_rejected(self):
        s=self.session()
        for value in (float('nan'),float('inf'),-1,True):
            with self.subTest(value=value),self.assertRaises(runtime.ExecutionDenied):
                s.record_event('resource_rate',{'rate':'t_reference'},value)

    def test_observed_cycle_is_committed_before_budget_stop_and_survives_restart(self):
        s=self.session()
        def stop():raise runtime.BudgetStop('synthetic post-measurement quota boundary')
        s.checkpoint=stop
        with self.assertRaises(runtime.BudgetStop):
            s.record_event('resource_rate',{'rate':'t_reference'},2.75)
        self.assertFalse(s.db.in_transaction)
        row=s.db.execute('SELECT name,metadata,seconds FROM events').fetchone()
        self.assertEqual(row,('resource_rate',json.dumps({'rate':'t_reference'},sort_keys=True),2.75))
        self.assertEqual(s.observed_rates['t_reference'],2.75)
        restarted=runtime.GuardedExecution('main','fold-0',None)
        restarted.db=s.db
        restarted.refresh_observed_rates()
        self.assertEqual(restarted.observed_rates['t_reference'],2.75)

    def test_inactive_session_cannot_record_events_even_with_noop_checkpoint(self):
        s=self.session();s.active=False
        with self.assertRaises(runtime.ExecutionDenied):
            s.record_event('resource_rate',{'rate':'t_reference'},1.)
        self.assertEqual(s.db.execute('SELECT COUNT(*) FROM events').fetchone()[0],0)

    def test_task_count_is_loaded_once_and_maintained_for_success_and_failure(self):
        s=self.session();statements=[]
        s.db.set_trace_callback(statements.append)
        for index in range(18):
            if index%3==0:
                with self.assertRaises(RuntimeError):
                    with s.operation('encoder',{'phase':'synthetic_setup'}):raise RuntimeError('synthetic')
            else:
                with s.operation('encoder',{'phase':'synthetic_setup'}):pass
            self.assertEqual(s._count_task('encoder'),index+1)
        scans=[sql for sql in statements if sql.upper().startswith('SELECT') and 'COUNT(' in sql.upper() and 'OPERATIONS' in sql.upper()]
        self.assertEqual(len(scans),1,scans)
        self.assertEqual(s.task_counts['encoder'],18)
        self.assertEqual(s.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0],18)
        s.task_counts=None
        self.assertEqual(s._count_task('encoder'),18)

    def test_recorded_operation_duration_equals_live_and_reconstructed_maxima(self):
        for kind in ('encoder','kmeans'):
            s=self.session()
            with self.subTest(kind=kind),patch.object(runtime.time,'perf_counter',side_effect=[100.,100.125]):
                with s.operation(kind,{'phase':'synthetic_setup'}):pass
            stored=s.db.execute('SELECT seconds FROM operations').fetchone()[0]
            self.assertEqual(stored,.125)
            self.assertEqual(s.observed_rates['t_'+kind],stored)
            restarted=runtime.GuardedExecution('main','fold-0',None);restarted.db=s.db
            restarted.refresh_observed_rates()
            self.assertEqual(restarted.observed_rates['t_'+kind],stored)

    def test_remaining_projection_does_not_rescan_growing_operations_or_events(self):
        s=self.session();rates={'t_utility':.1,'t_full_candidate':.1,'t_reference':.2,'t_encoder':.01,
                             't_kmeans':.02,'t_game_setup':.03,'t_selector_setup':.04}
        s.gate={'measured_parameters':rates}
        s.observed_rates={'t_reference':.3}
        statements=[];s.db.set_trace_callback(statements.append)
        before=s.remaining_projection_seconds();after=s.remaining_projection_seconds()
        self.assertEqual(before,after)
        self.assertFalse([sql for sql in statements if 'OPERATIONS' in sql.upper() or 'EVENTS' in sql.upper()],statements)

    def test_unauthorized_entry_never_opens_scientific_sqlite(self):
        with patch.object(runtime.sqlite3,'connect',side_effect=AssertionError('disk SQLite forbidden')):
            with self.assertRaises(runtime.ExecutionDenied):runtime.preflight('probe',self.tasks[0]['task_id'],None)


if __name__ == '__main__':
    unittest.main()
