import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.contracts import TrainingPartition
from src.preprocessing import TabularEncoder
from src.samplers import select_rows,cluster_quotas,rank_indices
from src.evaluation import classification_metrics
from src.data import load_dataset,seed_for

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.X=pd.DataFrame({'x':[0.,0.,1.,2.,3.,4.,10.,10.,11.,12.,13.,14.], 'cat':['a','b']*6})
        self.train=TrainingPartition(tuple(f'r{i:02}' for i in range(12)),(0,)*6+(1,)*6,
            tuple(self.X.columns),tuple(f'g{i}' for i in range(12)),'train','toy')

    def test_encoder_fits_training_only(self):
        enc=TabularEncoder(['x'],['cat']).fit(self.X.iloc[:4])
        before=repr(enc.params_)
        out=enc.transform(pd.DataFrame({'x':[10000.,np.nan],'cat':['unseen','a']}))
        self.assertEqual(before,repr(enc.params_))
        self.assertEqual(enc.params_['x']['impute'],.5)
        self.assertEqual(enc.params_['x']['center'],.75)
        self.assertTrue(np.all(out[0,enc.blocks_['cat']]==0))

    def test_all_missing_and_constant_columns_preserved(self):
        X=pd.DataFrame({'x':[np.nan]*4,'cat':[None]*4})
        enc=TabularEncoder(['x'],['cat']).fit(X)
        self.assertEqual(enc.transform(X).shape,(4,2))
        self.assertTrue(np.isfinite(enc.transform(X)).all())
        self.assertEqual(enc.params_['cat']['categories'],['__MISSING__'])

    def test_robust_geometry_clips_only_geometry(self):
        X=pd.DataFrame({'x':list(range(99))+[100000.]})
        enc=TabularEncoder(['x'],[],'robust_geometry').fit(X)
        self.assertLess(enc.params_['x']['clip_high'],100000.)
        self.assertEqual(X.x.iloc[-1],100000.)
        self.assertEqual(enc.params_['x']['center'],49.5)

    def test_categorical_block_normalization(self):
        enc=TabularEncoder(['x'],['cat'],'ordinary_geometry').fit(self.X)
        a=enc.transform(self.X)
        np.testing.assert_allclose(np.sum(a[:,enc.blocks_['cat']]**2,axis=1),.5)

    def test_original_block_map_covers_once(self):
        enc=TabularEncoder(['x'],['cat']).fit(self.X)
        columns=[i for b in enc.blocks_.values() for i in b]
        self.assertEqual(columns,list(range(3)))

    def test_samplers_have_real_unique_balanced_rows(self):
        with threadpool_limits(limits=1):
            for strategy in ('stratified_random','ordinary_geometry','robust_geometry'):
                ix,res,diag=select_rows(self.X,self.train,['x'],['cat'],.5,17,strategy)
                self.assertEqual(len(ix),6);self.assertEqual(len(set(ix)),6)
                self.assertEqual(res.class_counts,{0:3,1:3})
                self.assertEqual(res.validate(self.train),.5)

    def test_all_strategies_full_identity(self):
        for strategy in ('stratified_random','ordinary_geometry','robust_geometry'):
            ix,res,diag=select_rows(self.X,self.train,['x'],['cat'],1.,17,strategy)
            np.testing.assert_array_equal(ix,np.arange(12))
            self.assertIsNone(diag['geometry'])

    def test_seed_reproducibility(self):
        with threadpool_limits(limits=1):
            for strategy in ('stratified_random','ordinary_geometry','robust_geometry'):
                a=select_rows(self.X,self.train,['x'],['cat'],.5,17,strategy)[0]
                b=select_rows(self.X,self.train,['x'],['cat'],.5,17,strategy)[0]
                np.testing.assert_array_equal(a,b)

    def test_geometry_duplicate_representations(self):
        X=pd.DataFrame({'x':[1.]*12,'cat':['a']*12})
        with threadpool_limits(limits=1):
            ix,res,diag=select_rows(X,self.train,['x'],['cat'],.5,17,'robust_geometry')
        self.assertEqual(len(ix),6)
        self.assertTrue(all(c['effective_clusters']==1 for c in diag['clusters']))

    def test_cluster_capacity_invariants(self):
        rng=np.random.default_rng(18)
        for _ in range(80):
            sizes=rng.integers(1,50,size=int(rng.integers(1,11)))
            for target in (len(sizes),int(sizes.sum()),int(rng.integers(len(sizes),sizes.sum()+1))):
                q=cluster_quotas(sizes,target)
                self.assertEqual(q.sum(),target);self.assertTrue(np.all((q>=1)&(q<=sizes)))

    def test_rank_positions_unique_and_in_range(self):
        for n in range(1,30):
            for q in range(1,n+1):
                r=rank_indices(n,q)
                self.assertEqual(len(set(r)),q)
                self.assertTrue(np.all((r>=0)&(r<n)))

    def test_metric_hand_calculation(self):
        m=classification_metrics([0,0,1,1],[.1,.8,.7,.9])
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(1,1,0,2))
        self.assertEqual(m['f1'],.8);self.assertEqual(m['roc_auc'],.75)

    def test_zero_denominator_and_single_class(self):
        m=classification_metrics([0,0],[.1,.2])
        self.assertTrue(m['f1_undefined']);self.assertEqual(m['f1'],0)
        self.assertIsNone(m['roc_auc']);self.assertIsNone(m['recall'])

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError): classification_metrics([0,1],[np.nan,.1])

    def test_official_adult_test_rejected(self):
        with self.assertRaises(ValueError):load_dataset('adult_test')

    def test_seeds_role_specific_and_stable(self):
        self.assertEqual(seed_for(1,'x','fold'),seed_for(1,'x','fold'))
        self.assertNotEqual(seed_for(1,'x','fold'),seed_for(1,'x','sampler'))

if __name__=='__main__':unittest.main()
