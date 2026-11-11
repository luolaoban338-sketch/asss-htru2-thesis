"""Fail-closed execution controls for the isolated Adult stage.

Import and preflight are read-only. A ledger is opened only inside a verified,
explicitly authorized execution context. No authorization is supplied by this
no-fit preparation delivery.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import sqlite3
import time
import hmac,secrets

ROOT=Path(__file__).resolve().parent
KINDS=('classifier','encoder','kmeans')
ROLES=('probe','main','replay','audit')

class ExecutionDenied(RuntimeError):pass
class BudgetStop(ExecutionDenied):pass

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def parse_utc(value):
    dt=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if dt.tzinfo is None:raise ExecutionDenied('UTC offset is required')
    return dt.astimezone(timezone.utc)

def finite_number(value):return type(value) in (int,float) and math.isfinite(value)

def prior_probe_usage():
    """Reverify sealed audited v1/v2 receipts without connecting to either ledger."""
    prior=read_json(ROOT/'prior_probe_usage.json')
    for path,expected in prior['source_sha256'].items():
        if sha256(path)!=expected:raise ExecutionDenied('Preserved prior evidence changed: '+path)
        if str(path).endswith('.sqlite3') and any(Path(str(path)+suffix).exists() for suffix in ('-wal','-shm','-journal')):
            raise ExecutionDenied('Prior ledger is not sealed')
    state=read_json(ROOT.parent/'thesis_adult_reduced_20261108_v2/state.json')
    if prior['counts']!=state['combined_operation_counts'] or prior['session_seconds']!=state['combined_probe_charged_seconds']:
        raise ExecutionDenied('Prior usage does not reconcile with audited delivery')
    if prior['old_gate_status']!='failed_runtime_feasibility' or state['runtime_gate_passed'] is not False:
        raise ExecutionDenied('Old failed gate status changed')
    return prior


def validate_paths(paths, *, writable=False):
    """No writes here; write-probe is an explicit no-fit preparation command."""
    store=Path('F:/ChatGPT/论文大数据').resolve()
    dirs={'data_root':'datasets','run_root':'runs','model_root':'models','scratch_root':'scratch','archive_root':'archives'}
    for key,kind in dirs.items():
        expected=store/kind/('adult_reduced_20261107' if key=='data_root' else 'adult_exact_20261110')
        actual=Path(paths[key]).resolve()
        if actual!=expected or not actual.is_relative_to(store):raise ExecutionDenied('Invalid F storage path: '+key)
        if writable and (not actual.is_dir() or not os.access(actual,os.W_OK)):raise ExecutionDenied('F storage unavailable: '+key)
    if Path(paths['stage_root']).resolve()!=ROOT:raise ExecutionDenied('Stage root mismatch')
    if Path(paths['stop_file']).resolve()!=ROOT/'STOP_REQUESTED':raise ExecutionDenied('STOP path mismatch')
    for key,parent in [('ledger_file','run_root'),('runtime_gate_file','run_root'),('partitions_file','data_root'),('games_file','data_root'),('timing_tasks_file','data_root')]:
        if not Path(paths[key]).resolve().is_relative_to(Path(paths[parent]).resolve()):raise ExecutionDenied('Artifact path escapes storage: '+key)
    return True

def validate_authorization(auth,config,paths,role,task,freeze_sha,*,now=None):
    now=now or datetime.now(timezone.utc)
    if role not in ROLES or not task or any(c in task for c in '/\\:.'):raise ExecutionDenied('Invalid role/task')
    required={'explicit_execution_authorized':True,'role':role,'task':task,'freeze_sha256':freeze_sha,
              'config_sha256':sha256(ROOT/'config.json'),'paths_sha256':sha256(ROOT/'paths.json')}
    if auth.get('explicit_execution_authorized') is not True or any(auth.get(k)!=v for k,v in required.items()):raise ExecutionDenied('Missing or mismatched explicit execution authorization')
    if not auth.get('authorization_id') or not str(auth.get('user_instruction','')).strip():raise ExecutionDenied('Missing authorization provenance')
    if not parse_utc(auth['issued_at'])<=now<parse_utc(auth['expires_at']):raise ExecutionDenied('Authorization expired or not yet valid')
    if now-parse_utc(auth['issued_at'])>__import__('datetime').timedelta(hours=24):raise ExecutionDenied('Authorization requires a fresh entry')
    values=[auth.get(k) for k in ('entry_used_percent','stop_used_percent','ceiling_used_percent')]
    if not all(finite_number(x) for x in values):raise ExecutionDenied('Missing finite account envelope')
    entry,stop,ceiling=values
    if not 0<=entry<stop<=ceiling<=100-config['account_protected_points']:raise ExecutionDenied('Invalid envelope or protected reserve')
    caps=auth.get('attempt_caps',{})
    role_caps=config['budget']['future_timing_probe']['hard_caps'] if role=='probe' else config['budget']['execution_role_caps'][role]
    if any(type(caps.get(k)) is not int or not 0<caps[k]<=role_caps[k] for k in KINDS):raise ExecutionDenied('Invalid finite task caps')
    if not auth.get('quota_observation_file'):raise ExecutionDenied('Fresh quota observation file required')
    return True

def validate_quota(auth,*,now=None):
    now=now or datetime.now(timezone.utc)
    q=read_json(auth['quota_observation_file'])
    if q.get('authorization_id')!=auth['authorization_id']:raise BudgetStop('Quota observation belongs to another authorization')
    age=(now-parse_utc(q['observed_at'])).total_seconds()
    if not 0<=age<=300:raise BudgetStop('Quota observation stale or future-dated')
    value=q.get('used_percent')
    if not finite_number(value) or not 0<=value<=100:raise BudgetStop('Invalid quota observation')
    if value>=auth['stop_used_percent']:raise BudgetStop('Authorized account stop reached')
    if value<auth['entry_used_percent']:raise BudgetStop('Account usage fell below entry; reset or changed observation requires new authorization')
    return value

def validate_gate(paths,freeze_sha):
    path=Path(paths['runtime_gate_file'])
    if not path.is_file():raise ExecutionDenied('Adult runtime gate has not been measured')
    gate=read_json(path)
    if gate.get('status')!='passed' or gate.get('freeze_sha256')!=freeze_sha:raise ExecutionDenied('Runtime gate not passed for this freeze')
    if not finite_number(gate.get('projected_total_seconds')) or not 0<gate['projected_total_seconds']<=172800:raise ExecutionDenied('Invalid runtime projection')
    if gate.get('timing_tasks_complete')!=75 or gate.get('independent_checks_passed') is not True:raise ExecutionDenied('Incomplete timing evidence')
    if not gate.get('source_sha256'):raise ExecutionDenied('Gate lacks timing source hashes')
    root=Path(paths['run_root']).resolve()
    for rel,digest in gate['source_sha256'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or sha256(p)!=digest:raise ExecutionDenied('Timing evidence changed')
    rates=gate.get('measured_parameters',{})
    for name in ('t_utility','t_full_candidate','t_reference','t_encoder','t_kmeans','t_game_setup','t_selector_setup','T_probe_actual'):
        if not finite_number(rates.get(name)) or rates[name]<=0:raise ExecutionDenied('Missing or nonfinite timing measurement: '+name)
    from gate import verify_published_gate
    verify_published_gate(paths,freeze_sha,gate)
    return gate

def preflight(role,task,authorization_file):
    config=read_json(ROOT/'config.json');paths=read_json(ROOT/'paths.json')
    if authorization_file is None:raise ExecutionDenied('Execution authorization file is required; no-fit preparation does not authorize fits')
    if Path(paths['stop_file']).exists():raise ExecutionDenied('STOP_REQUESTED is set')
    from freeze import verify_freeze
    frozen=verify_freeze()
    validate_paths(paths,writable=True)
    prior_probe_usage()
    auth=read_json(authorization_file)
    validate_authorization(auth,config,paths,role,task,frozen['freeze_sha256'])
    if role in ('main','replay'):
        permitted={f'fold-{i}' for i in (range(5) if role=='main' else [0])}
    elif role=='audit':permitted={f'main-fold-{i}' for i in range(5)}|{'replay-fold-0'}
    else:permitted={x['task_id'] for x in read_json(paths['timing_tasks_file'])['tasks']}
    if task not in permitted:raise ExecutionDenied('Task is outside the frozen schedule')
    validate_quota(auth)
    gate=None if role=='probe' else validate_gate(paths,frozen['freeze_sha256'])
    return config,paths,auth,frozen,gate

class GuardedExecution:
    def __init__(self,role,task,authorization_file):
        self.role,self.task,self.authorization_file=role,task,authorization_file
        self.gate=None;self.observed_rates={};self.binding_key=secrets.token_bytes(32);self.task_counts=None
        self.active=False;self.db=None;self.lock=None;self.started=None;self.session_id=None
    def __enter__(self):
        self.config,self.paths,self.auth,self.freeze,self.gate=preflight(self.role,self.task,self.authorization_file)
        import msvcrt
        self.lock=Path(self.paths['run_root'],'execution.lock').open('a+b')
        self.lock.seek(0);self.lock.write(b'0');self.lock.flush();self.lock.seek(0)
        try:
            msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            self.db=sqlite3.connect(self.paths['ledger_file'],timeout=30)
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS operations(id INTEGER PRIMARY KEY,session_id INTEGER,role TEXT,task TEXT,kind TEXT,phase TEXT,metadata TEXT,status TEXT,seconds REAL,error TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY,role TEXT,task TEXT,status TEXT,seconds REAL,authorization_id TEXT,started_utc TEXT);
CREATE TABLE IF NOT EXISTS counters(role TEXT,kind TEXT,n INTEGER,PRIMARY KEY(role,kind));
CREATE INDEX IF NOT EXISTS operations_task_kind ON operations(role,task,kind);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,session_id INTEGER,name TEXT,metadata TEXT,seconds REAL);
CREATE TABLE IF NOT EXISTS type_counters(role TEXT,bucket TEXT,n INTEGER,PRIMARY KEY(role,bucket));
CREATE TABLE IF NOT EXISTS completion_evidence(role TEXT,task TEXT,path TEXT,sha256 TEXT,PRIMARY KEY(role,task));
CREATE TABLE IF NOT EXISTS completed_tasks(role TEXT,task TEXT,PRIMARY KEY(role,task));''')
            previous=self.db.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone()
            if previous and previous[0]!=self.freeze['freeze_sha256']:raise ExecutionDenied('Ledger freeze changed')
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('freeze_sha256',?)",(self.freeze['freeze_sha256'],));self.db.commit()
            if self.db.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0] or self.db.execute("SELECT COUNT(*) FROM operations WHERE status='started'").fetchone()[0]:raise ExecutionDenied('Interrupted operation requires a separate recovery audit; no automatic reset')
            expected={(r,k):n for r,k,n in self.db.execute('SELECT role,kind,COUNT(*) FROM operations GROUP BY role,kind')}
            actual={(r,k):n for r,k,n in self.db.execute('SELECT role,kind,n FROM counters')}
            if expected!=actual:raise ExecutionDenied('Counters do not reconcile')
            if self.db.execute('SELECT 1 FROM completed_tasks WHERE role=? AND task=?',(self.role,self.task)).fetchone():raise ExecutionDenied('Task already completed; use read-only verification')
            expected_types={}
            for role,metadata in self.db.execute("SELECT role,metadata FROM operations WHERE kind='classifier'"):
                bucket=json.loads(metadata).get('classifier_bucket')
                if bucket not in ('pool','reference'):raise ExecutionDenied('Missing recorded classifier type')
                expected_types[role,bucket]=expected_types.get((role,bucket),0)+1
            if expected_types!={(r,b):n for r,b,n in self.db.execute('SELECT role,bucket,n FROM type_counters')}:
                raise ExecutionDenied('Classifier type counters do not reconcile')
            self.verify_completion_evidence()
            from partitions import load_partitions,load_timing_tasks
            from resource_binding import MembershipBinding
            schema=read_json(self.config['data_source']['dictionary'])
            self.membership=MembershipBinding(load_partitions(),load_timing_tasks(),[r['field'] for r in schema if r['role']=='predictor'])
            self.refresh_observed_rates()
            inherited_seconds=prior_probe_usage()['session_seconds']
            self.prior_seconds=inherited_seconds+self.db.execute('SELECT COALESCE(SUM(seconds),0) FROM sessions').fetchone()[0]
            self.prior_probe_seconds=self.db.execute("SELECT COALESCE(SUM(seconds),0) FROM sessions WHERE role='probe'").fetchone()[0]
            self.started=time.perf_counter();self.active=True
            self.checkpoint()
            if self.gate and self.prior_seconds+self.remaining_projection_seconds()>172800:raise BudgetStop('Remaining conservative projection exceeds cumulative 48h')
            self.session_id=self.db.execute("INSERT INTO sessions(role,task,status,seconds,authorization_id,started_utc) VALUES (?,?,'active',0,?,?)",(self.role,self.task,self.auth['authorization_id'],datetime.now(timezone.utc).isoformat())).lastrowid
            self.db.commit()
            self.initial_task_counts={k:self._count_task(k) for k in KINDS}
            return self
        except BaseException:
            self.active=False;self._close();raise
    def _count(self,role,kind):
        row=self.db.execute('SELECT n FROM counters WHERE role=? AND kind=?',(role,kind)).fetchone();return row[0] if row else 0
    def _count_task(self,kind):
        if self.task_counts is None:
            self.task_counts={k:0 for k in KINDS}
            self.task_counts.update(dict(self.db.execute('SELECT kind,COUNT(*) FROM operations WHERE role=? AND task=? GROUP BY kind',(self.role,self.task))))
        return self.task_counts[kind]
    def require_execution(self):
        if not self.active or self.db is None:raise ExecutionDenied('An active verified execution session is required')
        self.checkpoint()
    def checkpoint(self):
        if not self.active:raise ExecutionDenied('Execution session inactive')
        if Path(self.paths['stop_file']).exists():raise BudgetStop('STOP_REQUESTED is set')
        now=datetime.now(timezone.utc)
        if now>=parse_utc(self.auth['expires_at']):raise BudgetStop('Authorization expired')
        validate_quota(self.auth,now=now)
        elapsed=time.perf_counter()-self.started
        if self.prior_seconds+elapsed>=172800:raise BudgetStop('Cumulative 48-hour cap reached')
        invocation=1800 if self.role=='probe' else 3600
        if elapsed>=invocation:raise BudgetStop('Finite invocation cap reached')
        if self.role=='probe' and self.prior_probe_seconds+elapsed>=7200:raise BudgetStop('Probe 2-hour cap reached')
        if self.gate and self.prior_seconds+elapsed+self.remaining_projection_seconds()>172800:raise BudgetStop('Updated remaining projection exceeds cumulative 48h')
    def artifact_path(self,relative):
        base=Path(self.paths['run_root'])/self.role/self.task
        if not base.resolve().is_relative_to(Path(self.paths['run_root']).resolve()):raise ExecutionDenied('Task storage resolves outside F run root')
        path=(base/relative).resolve()
        if not path.is_relative_to(base.resolve()):raise ExecutionDenied('Artifact path escapes current task')
        return path
    @contextmanager
    def operation(self,kind,metadata):
        self.require_execution()
        if kind not in KINDS or not isinstance(metadata,dict) or 'phase' not in metadata:raise ExecutionDenied('Unknown operation kind or phase')
        if metadata.get('role',self.role)!=self.role or metadata.get('task',self.task)!=self.task:raise ExecutionDenied('Operation role/task mismatch')
        if kind=='classifier':self.verify_classifier_binding(metadata)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            budget=self.config['budget']
            for k in KINDS:
                # All operation types reserve capacity for replay and independent audit.
                if k!=kind:continue
                rolecap=budget['future_timing_probe']['hard_caps'][k] if self.role=='probe' else budget['execution_role_caps'][self.role][k]
                if self._count(self.role,k)>=rolecap:raise BudgetStop('Role attempt cap reached: '+k)
                total=sum(self._count(r,k) for r in ROLES)
                if total>=budget['total_future_caps_including_probe'][k]:raise BudgetStop('Combined attempt cap reached: '+k)
                if self._count_task(k)-self.initial_task_counts[k]>=self.auth['attempt_caps'][k]:raise BudgetStop('Authorized finite invocation attempt cap reached: '+k)
            if kind=='classifier':
                bucket=metadata['classifier_bucket']
                if self.role!='probe':
                    cap=budget['classifier_buckets_by_role'][self.role][bucket]['total_cap']
                    row=self.db.execute('SELECT n FROM type_counters WHERE role=? AND bucket=?',(self.role,bucket)).fetchone()
                    if row and row[0]>=cap:raise BudgetStop('Classifier type attempt cap reached: '+bucket)
                self.db.execute('INSERT INTO type_counters VALUES (?,?,1) ON CONFLICT(role,bucket) DO UPDATE SET n=n+1',(self.role,bucket))
            op=self.db.execute("INSERT INTO operations(session_id,role,task,kind,phase,metadata,status) VALUES (?,?,?,?,?,?,'started')",(self.session_id,self.role,self.task,kind,metadata['phase'],json.dumps(metadata,sort_keys=True))).lastrowid
            self.db.execute('INSERT INTO counters VALUES (?,?,1) ON CONFLICT(role,kind) DO UPDATE SET n=n+1',(self.role,kind));self.db.commit()
            self.task_counts[kind]+=1
        except BaseException:self.db.rollback();raise
        start=time.perf_counter()
        try:yield op
        except BaseException as exc:
            measured=time.perf_counter()-start
            self.db.execute("UPDATE operations SET status='failed',seconds=?,error=? WHERE id=?",(measured,repr(exc),op));self.db.commit();raise
        else:
            measured=time.perf_counter()-start
            self.db.execute("UPDATE operations SET status='success',seconds=? WHERE id=?",(measured,op));self.db.commit()
        finally:
            if kind in ('encoder','kmeans'):
                key='t_'+kind;self.observed_rates[key]=max(self.observed_rates.get(key,0),measured)
    def record_event(self,name,metadata,seconds):
        # Recording already-observed work is allowed before a cooperative stop.
        # Persist first so a resource stop cannot discard the larger observation.
        if not self.active or self.db is None:raise ExecutionDenied('Execution session inactive')
        if not finite_number(seconds) or seconds<0:raise ExecutionDenied('Invalid event duration')
        self._observe_event(name,metadata,seconds)
        row=self.db.execute('INSERT INTO events(session_id,name,metadata,seconds) VALUES (?,?,?,?)',(self.session_id,name,json.dumps(metadata,sort_keys=True),seconds));self.db.commit()
        self.checkpoint();return row.lastrowid
    def bind_classifier(self,selected,evaluation,features,metadata):
        self.require_execution()
        try:bound=self.membership.classify(self.role,self.task,selected,evaluation,features,metadata)
        except (ValueError,KeyError,StopIteration,IndexError) as exc:raise ExecutionDenied('Classifier membership binding rejected: '+str(exc)) from exc
        bound['binding_token']=self._binding_digest(bound)
        return bound
    def _binding_digest(self,metadata):
        payload=json.dumps({k:v for k,v in metadata.items() if k!='binding_token'},sort_keys=True,separators=(',',':')).encode()
        return hmac.new(self.binding_key,payload,'sha256').hexdigest()
    def verify_classifier_binding(self,metadata):
        if metadata.get('classifier_bucket') not in ('pool','reference') or not hmac.compare_digest(str(metadata.get('binding_token','')),self._binding_digest(metadata)):
            raise ExecutionDenied('Classifier lacks controlled membership binding')
    def refresh_observed_rates(self):
        for kind,key in [('encoder','t_encoder'),('kmeans','t_kmeans')]:
            value=self.db.execute('SELECT MAX(seconds) FROM operations WHERE kind=?',(kind,)).fetchone()[0]
            if value is not None:
                if not finite_number(value) or value<0:raise ExecutionDenied('Invalid observed operation duration')
                self.observed_rates[key]=max(self.observed_rates.get(key,0),value)
        for name,metadata,seconds in self.db.execute('SELECT name,metadata,seconds FROM events'):
            self._observe_event(name,json.loads(metadata),seconds)
    def _observe_event(self,name,metadata,seconds):
        if not finite_number(seconds) or seconds<0:raise ExecutionDenied('Invalid event duration')
        if name=='resource_rate':key=metadata['rate']
        elif name=='audit_complete_cycle':key='t_direct_audit_reference' if metadata.get('phase')=='audit_reference' else 't_direct_audit_cycle'
        else:return
        allowed={'t_utility','t_full_candidate','t_reference','t_encoder','t_kmeans','t_game_setup','t_selector_setup','t_direct_audit_cycle','t_direct_audit_reference'}
        if key not in allowed:raise ExecutionDenied('Unknown measured rate')
        self.observed_rates[key]=max(self.observed_rates.get(key,0),seconds)
    def remaining_projection_seconds(self):
        if self.gate is None:return 0
        from projection import remaining_seconds,refresh_rates
        try:
            rates=refresh_rates(self.gate['measured_parameters'],self.observed_rates.items())
            return remaining_seconds(rates,list(self.db.execute('SELECT role,task FROM completed_tasks')))
        except (ValueError,KeyError) as exc:raise ExecutionDenied(str(exc)) from exc
    def verify_completion_evidence(self):
        done=set(self.db.execute('SELECT role,task FROM completed_tasks'))
        from projection import remaining_counts
        try:remaining_counts(list(done))
        except ValueError as exc:raise ExecutionDenied(str(exc)) from exc
        evidence={(r,t):(p,h) for r,t,p,h in self.db.execute('SELECT role,task,path,sha256 FROM completion_evidence')}
        if done!=set(evidence):raise ExecutionDenied('Completion evidence missing or extraneous')
        for (role,task),(p,h) in evidence.items():
            base=(Path(self.paths['run_root'])/role/task).resolve();path=Path(p).resolve()
            expected='timing.json' if role=='probe' else ('audit.json' if role=='audit' else 'bundle.json')
            if path!=base/expected or sha256(path)!=h:raise ExecutionDenied('Completed package evidence changed')
            value=read_json(path)
            self.validate_completion_record(role,task,value,base)
            marker=path.with_suffix('.json.sha256')
            if not marker.is_file() or marker.read_text('ascii').strip()!=h:raise ExecutionDenied('Completion marker changed')
            for rel,wanted in value.get('artifact_sha256',{}).items():
                child=(base/rel).resolve()
                if not child.is_relative_to(base) or sha256(child)!=wanted:raise ExecutionDenied('Completed bundle changed')
    def validate_completion_record(self,role,task,value,base):
        if role=='probe':
            from projection import PROBE_TASKS
            if task not in PROBE_TASKS or value.get('status')!='complete' or value.get('task_id')!=task:
                raise ExecutionDenied('Incomplete or mismatched probe record')
        elif role in ('main','replay'):
            permitted={f'fold-{i}' for i in (range(5) if role=='main' else [0])}
            if task not in permitted or value.get('status')!='complete_pending_independent_audit' or value.get('role')!=role or value.get('fold')!=int(task[-1]):
                raise ExecutionDenied('Incomplete or mismatched computation bundle')
            artifacts=value.get('artifact_sha256',{})
            if not isinstance(artifacts,dict) or not {'outer_release.json','decision_lock.json','decisions.json'}<=set(artifacts):
                raise ExecutionDenied('Completed bundle lacks required artifact evidence')
            for rel,wanted in artifacts.items():
                child=(base/rel).resolve()
                if not child.is_relative_to(base) or sha256(child)!=wanted:raise ExecutionDenied('Completed bundle artifact changed')
        elif role=='audit':
            permitted={f'main-fold-{i}' for i in range(5)}|{'replay-fold-0'}
            if task not in permitted:raise ExecutionDenied('Unknown audit completion')
            source,_,fold=task.split('-')
            source_path=Path(self.paths['run_root'])/source/('fold-'+fold)/'bundle.json'
            if value.get('status')!='passed' or value.get('source_role')!=source or value.get('fold')!=int(fold) or value.get('source_bundle_sha256')!=sha256(source_path):
                raise ExecutionDenied('Incomplete or mismatched direct audit')
        else:raise ExecutionDenied('Unknown completion role')
    def mark_completed(self):
        self.require_execution()
        name='timing.json' if self.role=='probe' else ('audit.json' if self.role=='audit' else 'bundle.json')
        path=self.artifact_path(name);value=read_json(path);digest=sha256(path)
        expected='complete' if self.role=='probe' else ('passed' if self.role=='audit' else 'complete_pending_independent_audit')
        if value.get('status')!=expected:raise ExecutionDenied('Incomplete package cannot earn projection credit')
        self.validate_completion_record(self.role,self.task,value,path.parent)
        if path.with_suffix('.json.sha256').read_text('ascii').strip()!=digest:raise ExecutionDenied('Package marker mismatch')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO completion_evidence VALUES (?,?,?,?)',(self.role,self.task,str(path),digest))
            self.db.execute('INSERT INTO completed_tasks VALUES (?,?)',(self.role,self.task))
            self.verify_completion_evidence();self.db.commit()
        except BaseException:self.db.rollback();raise
    def __exit__(self,exc_type,exc,tb):
        if self.db is not None and self.session_id is not None:
            status='complete' if exc is None else ('paused' if isinstance(exc,BudgetStop) else 'failed')
            self.db.execute('UPDATE sessions SET status=?,seconds=? WHERE id=?',(status,time.perf_counter()-self.started,self.session_id));self.db.commit()
        self.active=False;self._close();return False
    def _close(self):
        if self.db is not None:self.db.close();self.db=None
        if self.lock is not None:
            try:
                import msvcrt
                self.lock.seek(0);msvcrt.locking(self.lock.fileno(),msvcrt.LK_UNLCK,1)
            except OSError:pass
            self.lock.close();self.lock=None
