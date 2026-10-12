"""Pre-fit analytic gates for frozen fixed-budget implementation."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from experiments.ablation import coordinate,seed_ledger,variants,Budget
from experiments.shapley import corrected_scores
from experiments.feature_path import forward_path
from scripts.run_joint_pilot import read,commit,cached
from scripts.run_ablation import ROOT,DESIGN,training_positions
from src.data import load_dataset
from src.contracts import assert_disjoint

CFG=read(DESIGN/'config.json')


class AblationGates(unittest.TestCase):
    def test_seed_known_contract_and_types(self):
        key,value=coordinate(CFG,'predictive',2026100501,0,'shared','game',3)
        raw='["ablation-htru2-fixedbudget-v1-20261005","predictive",2026100501,0,"shared","game",3]'
        self.assertEqual(value,int.from_bytes(hashlib.sha256(raw.encode()).digest()[:4],'big')%2147483647)
        self.assertEqual(key,json.loads(raw))
        for bad in (True,'0',0.0):
            with self.assertRaises(ValueError): coordinate(CFG,'predictive',2026100501,bad,'shared','game',0)

    def test_seed_ledger_complete_and_noncolliding(self):
        ledger=seed_ledger(CFG)
        self.assertEqual(len(ledger['top_level']),625)
        self.assertEqual(len(ledger['inherited_utility']),700)
        for records in (ledger['top_level'],ledger['inherited_utility']):
            self.assertEqual(len(records),len({r['seed'] for r in records}))
        self.assertEqual(seed_ledger(CFG),ledger)
        for phase in ('predictive','stability'):
            vals=[coordinate(CFG,phase,2026100501,0,'random_half','permutation',b)[1] for b in range(4)]
            self.assertEqual(len(set(vals)),4)

    def test_analytic_variance_switches_and_aliases(self):
        fs=tuple('abcdefgh'); phi=np.array([[0]*8,[1]*8,[2]*8,[3]*8],float)
        mc=np.array([[2,1,0,0,0,0,0,0]]*4,float)
        stats,paths=variants(fs,phi,mc,np.zeros((8,8)),CFG)
        self.assertAlmostEqual(stats['observed_variance'][0],5/3)
        self.assertTrue(stats['clipped_to_zero'][0])
        self.assertAlmostEqual(paths['111']['scores'][0],1.5)
        self.assertAlmostEqual(paths['011']['scores'][0],1.5-.5*np.sqrt(5/3))
        self.assertAlmostEqual(paths['111']['scores'][1],1.5-.5*np.sqrt(2/3))
        np.testing.assert_array_equal(paths['101']['scores'],np.ones(8)*1.5)
        self.assertEqual(paths['001']['path'],paths['101']['path'])
        self.assertEqual(paths['000']['path'],paths['100']['path'])

    def test_ties_negative_increments_and_redundancy(self):
        fs=tuple('abcdefgh'); phi=np.tile([-1.,-1.,-1.,-1.,-2.,-2.,-2.,-2.],(4,1))
        _,p=variants(fs,phi,np.zeros_like(phi),np.ones((8,8))-np.eye(8),CFG)
        self.assertEqual(p['111']['path'][-1]['features'],list(fs))
        self.assertEqual(p['111']['features'],list('abcd'))
        self.assertTrue(all(s['increment']<0 for s in p['111']['path']))
        scores=np.array([.10,.095,.09,0,0,0,0,0]); R=np.zeros((8,8)); R[0,1]=R[1,0]=1
        _,p=variants(fs,np.tile(scores,(4,1)),np.zeros((4,8)),R,CFG)
        self.assertEqual(p['111']['path'][1]['added'],'c')
        self.assertEqual(p['110']['path'][1]['added'],'b')

    def test_all_enabled_inherited_saved_arrays(self):
        folder=ROOT/'runs/joint-htru2-pilot-20261001-v1/fold-0/refit/random_half'
        old=read(folder/'selector.json'); games=[read(folder/f'game-{b}.json') for b in range(4)]
        phi=np.array([g['permutation']['phi'] for g in games]); mc=np.array([g['permutation']['mc_variance_of_mean'] for g in games])
        stats,paths=variants(old['features'],phi,mc,np.array(old['redundancy']),CFG)
        inherited=corrected_scores(phi,mc,.5)
        for k in inherited: np.testing.assert_array_equal(stats[k],inherited[k])
        self.assertEqual(paths['111']['path'],forward_path(old['features'],inherited['score'],old['redundancy'],.1))

    def test_durable_budget_failed_and_interrupted_attempts_count(self):
        with tempfile.TemporaryDirectory() as directory:
            db=Budget(Path(directory)/'budget.db',cap=3,audit_cap=1)
            n=db.reserve('logistic'); db.finish(n,'failed',.1,'deliberate test')
            db.reserve('logistic')  # Interrupted reservation remains charged.
            with db.track('audit:test'):
                n=db.reserve('logistic'); db.finish(n,'success',.1)
                with self.assertRaises(RuntimeError): db.reserve('logistic')
            with self.assertRaises(RuntimeError): db.reserve('logistic')
            self.assertEqual(sum(r['attempts'] for r in db.counts()),3)
            db.db.close()

    def test_resume_integrity_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json'; commit(path,{'x':1})
            self.assertEqual(cached(path),{'x':1})
            path.write_text('{"x":2}',encoding='utf-8')
            with self.assertRaises(ValueError): cached(path)

    def test_design_counts(self):
        self.assertEqual(5*3*3*4,CFG['main_schedule']['predictive_games'])
        self.assertEqual(10*3*4,CFG['main_schedule']['stability_games'])
        self.assertEqual(15*(3*(8+2)+1),CFG['main_schedule']['predictive_logical_records'])
        self.assertEqual((180+120+24)*765+375+25+10000,CFG['planned_logistic_fit_upper_bound'])
        self.assertLessEqual(CFG['planned_logistic_fit_upper_bound'],CFG['hard_logistic_fit_attempt_cap'])

    def test_training_only_resamples(self):
        data=load_dataset('htru2'); splits=read(ROOT/f"runs/{CFG['source_baseline_run']}/htru2/splits.json")
        self.assertEqual(len(set(data.group_ids)),len(data.y))
        from math import ceil
        for unit in range(10):
            train,test=training_positions(data,splits,CFG,'stability',2026100501,unit)
            self.assertEqual(len(train),ceil(.8*len(splits['folds'][0]['train_positions'])))
            assert_disjoint(data.partition(train,'gate'),data.row_ids[test],data.group_ids[test])
            self.assertTrue(set(train)<=set(splits['folds'][0]['train_positions']))

if __name__=='__main__': unittest.main()
