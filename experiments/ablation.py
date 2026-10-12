"""Frozen fixed-budget CSR adapters, deterministic coordinates, durable accounting."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from src.preprocessing import TabularEncoder
from src.data import seed_for
from experiments.shapley import corrected_scores
from experiments.feature_path import forward_path


def coordinate(config, phase, root, unit, sampler, role, index=0):
    if phase not in ('predictive', 'stability') or type(root) is not int or type(unit) is not int or type(index) is not int:
        raise ValueError('Invalid coordinate types')
    if sampler not in [*config['samplers'], 'shared']:
        raise ValueError('Invalid sampler')
    if role not in ('data_resample','final_rows','contribution_subsample','game','permutation','final_model','random_features'):
        raise ValueError('Unknown role')
    key = [config['protocol_id'], phase, root, unit, sampler, role, index]
    raw = json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return key, int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big') % 2147483647


def seed(config, phase, root, unit, sampler, role, index=0):
    return coordinate(config, phase, root, unit, sampler, role, index)[1]


def seed_ledger(config):
    top, inner = {}, {}
    units = [('predictive', root, f) for f in config['outer_folds'] for root in config['predictive_root_seeds']]
    units += [('stability', config['stability']['algorithm_root_seed'], i) for i in range(10)]
    def add(mapping, key, value):
        name = json.dumps(key, separators=(',', ':'))
        if value in mapping and mapping[value] != name:
            raise ValueError(f'Seed collision: {mapping[value]} and {name}')
        mapping[value] = name
    for phase, root, unit in units:
        keys = [('shared','final_model',0), ('shared','random_features',0)] if phase == 'predictive' else [('shared','random_features',0)]
        keys += [(s,'final_rows',0) for s in config['samplers']]
        for b in range(config['B']):
            keys += [('shared','contribution_subsample',b), ('shared','game',b)]
            keys += [(s,'permutation',b) for s in config['samplers']]
        for sampler, role, b in keys:
            key, value = coordinate(config, phase, root, unit, sampler, role, b)
            add(top, key, value)
            if role == 'game':
                add(inner, [value,'utility_split'], seed_for(value,'utility_split'))
                for f in range(config['utility_folds']):
                    for r in ('utility_sampler','utility_model'):
                        add(inner,[value,f,r],seed_for(value,f,r))
        if phase == 'stability':
            key, value = coordinate(config,phase,config['stability']['resample_root_seeds'][unit],unit,'shared','data_resample')
            add(top,key,value)
    return {'top_level':[{'key':json.loads(k),'seed':s} for s,k in sorted(top.items())],
            'inherited_utility':[{'key':json.loads(k),'seed':s} for s,k in sorted(inner.items())],
            'collision_check':'passed separately in both derivation namespaces'}


def variants(features, estimates, mc, redundancy, config):
    stats = corrected_scores(estimates, mc, config['lambda_enabled'])
    result = {}
    for label in config['logical_csr']:
        c,s,r = map(int,label)
        variance = np.maximum(0,stats['raw_corrected_variance']) if c else stats['observed_variance']
        score = stats['mean'] - (config['lambda_enabled'] * np.sqrt(variance) if s else 0)
        path = forward_path(features,score,redundancy,config['eta_enabled'] if r else 0)
        chosen = path[config['k']-1]['features']
        result[label] = {'scores':score, 'path':path, 'features':[f for f in features if f in chosen]}
    for alias, canonical in config['aliases'].items():
        assert result[alias]['path'] == result[canonical]['path']
    return stats, result


class Budget:
    """Reserve each attempt durably BEFORE calling fit, including interrupted fits.

    One worker takes an OS lock for its entire instrumentation context. SQLite is
    shared by main, replay and audits; no separate replay budget can hide attempts.
    """
    def __init__(self, path, cap=260000, audit_cap=10000):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.cap=cap; self.audit_cap=audit_cap
        self.db=sqlite3.connect(self.path,timeout=30)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS operations (id INTEGER PRIMARY KEY, kind TEXT, tag TEXT, status TEXT, seconds REAL, error TEXT)')
        self.db.commit()
        self.tag='unassigned'

    def reserve(self,kind):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if kind=='logistic':
                count=self.db.execute("SELECT COUNT(*) FROM operations WHERE kind='logistic'").fetchone()[0]
                if count>=self.cap: raise RuntimeError('All-stage logistic-fit attempt cap reached')
                if self.tag.startswith('audit:'):
                    audit=self.db.execute("SELECT COUNT(*) FROM operations WHERE kind='logistic' AND tag LIKE 'audit:%'").fetchone()[0]
                    if audit>=self.audit_cap: raise RuntimeError('Audit fit reserve exhausted')
            row=self.db.execute('INSERT INTO operations(kind,tag,status) VALUES (?,?,?)',(kind,self.tag,'started'))
            self.db.commit(); return row.lastrowid
        except BaseException:
            self.db.rollback(); raise

    def finish(self,number,status,seconds,error=None):
        self.db.execute('UPDATE operations SET status=?,seconds=?,error=? WHERE id=?',(status,seconds,error,number))
        self.db.commit()

    def counts(self):
        return [{'kind':k,'status':s,'attempts':n,'seconds':t or 0} for k,s,n,t in self.db.execute(
            'SELECT kind,status,COUNT(*),SUM(seconds) FROM operations GROUP BY kind,status ORDER BY kind,status')]

    @contextmanager
    def track(self,tag):
        prior=self.tag; self.tag=tag
        try: yield
        finally: self.tag=prior

    @contextmanager
    def instrument(self):
        import msvcrt
        lock=open(self.path.with_suffix('.lock'),'a+b')
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        originals=[]
        def wrapper(original,kind):
            def counted(obj,*args,**kwargs):
                number=self.reserve(kind); start=time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        result=original(obj,*args,**kwargs)
                except BaseException as exc:
                    self.finish(number,'failed',time.perf_counter()-start,repr(exc)); raise
                self.finish(number,'success',time.perf_counter()-start)
                return result
            return counted
        try:
            for cls,kind in [(LogisticRegression,'logistic'),(KMeans,'kmeans'),(TabularEncoder,'encoder')]:
                originals.append((cls,cls.fit)); cls.fit=wrapper(cls.fit,kind)
            yield self
        finally:
            for cls,original in originals: cls.fit=original
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()
