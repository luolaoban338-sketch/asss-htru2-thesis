"""Durable pre-fit accounting and bounded, single-worker execution."""
from contextlib import contextmanager
from pathlib import Path
import json, sqlite3, time, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from src.preprocessing import TabularEncoder

class BudgetStop(RuntimeError):
    pass

class Budget:
    def __init__(self, path, limits, stop_path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = dict(limits); self.stop_path = Path(stop_path)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS operations (id INTEGER PRIMARY KEY,kind TEXT,tag TEXT,status TEXT,seconds REAL,error TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY,status TEXT,seconds REAL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS counters (key TEXT PRIMARY KEY, attempts INTEGER NOT NULL)')
        expected = json.dumps(self.limits, sort_keys=True)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='limits'").fetchone()
        if previous and previous[0] != expected:
            self.db.close()
            raise ValueError('Frozen budget changed')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('limits',?)", (expected,)); self.db.commit()
        # Reconcile once per open; reservations then use O(1) durable counters.
        expected_counts={k:n for k,n in self.db.execute('SELECT kind,COUNT(*) FROM operations GROUP BY kind')}
        expected_counts['classifiers']=sum(expected_counts.get(k,0) for k in ('logistic','random_forest'))
        for role in ('main','preflight','audit','replay'):
            expected_counts['role:'+role]=self.db.execute("SELECT COUNT(*) FROM operations WHERE kind IN ('logistic','random_forest') AND tag LIKE ?",(role+':%',)).fetchone()[0]
        stored=dict(self.db.execute('SELECT key,attempts FROM counters'))
        if stored and any(stored.get(k,0)!=v for k,v in expected_counts.items()):
            self.db.close();raise RuntimeError('Budget counters do not reconcile with operations')
        for key,value in expected_counts.items():
            self.db.execute('INSERT OR IGNORE INTO counters VALUES (?,?)',(key,value))
        self.db.commit()
        self.tag = 'unassigned'; self.started = None

    def count(self,key):
        result=self.db.execute('SELECT attempts FROM counters WHERE key=?',(key,)).fetchone()
        return result[0] if result else 0

    def check(self):
        if self.stop_path.exists(): raise BudgetStop('STOP_REQUESTED is set')
        elapsed = self.db.execute("SELECT COALESCE(SUM(seconds),0) FROM sessions WHERE status!='active'").fetchone()[0]
        current = time.perf_counter() - self.started if self.started is not None else 0
        if elapsed + current >= self.limits['wall_seconds']: raise BudgetStop('Cumulative wall-time cap reached')
        if current >= self.limits['invocation_seconds']: raise BudgetStop('Invocation wall-time cap reached')

    def reserve(self, kind):
        self.check(); self.db.execute('BEGIN IMMEDIATE')
        try:
            count = self.count(kind)
            cap = self.limits.get(kind + '_attempts')
            if cap is not None and count >= cap: raise BudgetStop(kind + ' attempt cap reached')
            if kind in ('logistic', 'random_forest'):
                classifiers = self.count('classifiers')
                role=self.tag.split(':',1)[0]
                protected=sum(max(0,self.limits.get(other+'_reserve_attempts',0)-self.count('role:'+other))
                              for other in ('audit','replay') if other!=role)
                if classifiers >= self.limits['classifier_attempts']-protected:
                    raise BudgetStop('Combined classifier cap or protected replay/audit reserve reached')
            row = self.db.execute("INSERT INTO operations(kind,tag,status) VALUES (?,?,'started')", (kind,self.tag))
            keys=[kind]
            if kind in ('logistic','random_forest'):keys+=['classifiers','role:'+self.tag.split(':',1)[0]]
            for key in keys:
                self.db.execute('INSERT INTO counters(key,attempts) VALUES (?,1) ON CONFLICT(key) DO UPDATE SET attempts=attempts+1',(key,))
            self.db.commit(); return row.lastrowid
        except BaseException:
            self.db.rollback(); raise

    def counts(self):
        return [dict(kind=k,status=s,attempts=n,seconds=t or 0.) for k,s,n,t in self.db.execute(
            'SELECT kind,status,COUNT(*),SUM(seconds) FROM operations GROUP BY kind,status ORDER BY kind,status')]

    @contextmanager
    def track(self, tag):
        previous=self.tag; self.tag=tag
        try: yield
        finally: self.tag=previous

    @contextmanager
    def instrument(self):
        import msvcrt
        lock=open(self.path.with_suffix('.lock'),'a+b'); lock.seek(0)
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        originals=[]; session=None; status='failed'
        try:
            if self.db.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0]:
                raise RuntimeError('Interrupted active session requires explicit recovery audit')
            self.check()
            self.started=time.perf_counter()
            session=self.db.execute("INSERT INTO sessions(status,seconds) VALUES ('active',0)").lastrowid; self.db.commit()
            def wrapper(original, kind):
                def counted(obj,*args,**kwargs):
                    number=self.reserve(kind); start=time.perf_counter()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('error',ConvergenceWarning)
                            result=original(obj,*args,**kwargs)
                    except BaseException as exc:
                        self.db.execute("UPDATE operations SET status='failed',seconds=?,error=? WHERE id=?",(time.perf_counter()-start,repr(exc),number)); self.db.commit(); raise
                    self.db.execute("UPDATE operations SET status='success',seconds=? WHERE id=?",(time.perf_counter()-start,number)); self.db.commit()
                    return result
                return counted
            for cls,kind in [(LogisticRegression,'logistic'),(RandomForestClassifier,'random_forest'),(KMeans,'kmeans'),(TabularEncoder,'encoder')]:
                originals.append((cls,cls.fit)); cls.fit=wrapper(cls.fit,kind)
            yield self
            self.check()
            status='complete'
        except BudgetStop:
            status='paused'; raise
        finally:
            for cls,original in originals: cls.fit=original
            if session is not None:
                self.db.execute('UPDATE sessions SET status=?,seconds=? WHERE id=?',(status,time.perf_counter()-self.started,session)); self.db.commit()
            self.started=None
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

    def close(self): self.db.close()
