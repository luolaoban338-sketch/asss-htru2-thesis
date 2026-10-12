import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.stability import stability_summary, selection_matrix
from scripts.run_stability_pilot import restrict_to_pool, durable, run_replicate
import tests.test_joint_selection as toy_fixture
from threadpoolctl import threadpool_limits
from src.data import digest


class StabilityTests(unittest.TestCase):
    def test_identical_proper_subsets(self):
        s = stability_summary([['a'],['a'],['a']],['a','b'])
        self.assertEqual(s['nogueira'],1); self.assertEqual(s['jaccard_mean'],1)
        self.assertEqual(s['pair_count'],3)

    def test_disjoint_sets_preserve_negative_value(self):
        s = stability_summary([['a'],['b']],['a','b'])
        self.assertEqual(s['nogueira'],-1); self.assertEqual(s['jaccard_mean'],0)

    def test_varying_sizes_hand_calculation(self):
        s = stability_summary([['a'],['a','b'],['c']],['a','b','c'])
        self.assertAlmostEqual(s['nogueira'],-.35)
        self.assertAlmostEqual(s['jaccard_mean'],1/6)
        self.assertEqual(s['sizes'],[1,2,1])

    def test_degenerate_conventions(self):
        for sets,status in [([[],[]],'all_empty'),([['a','b'],['a','b']],'all_full')]:
            s = stability_summary(sets,['a','b'])
            self.assertIsNone(s['nogueira']); self.assertEqual(s['nogueira_status'],status)
            self.assertEqual(s['jaccard_mean'],1)

    def test_invalid_universe_and_sets(self):
        for sets,features in [([['a']],['a']),([[],[]],[]),([[],[]],['a','a']),
                              ([['b'],[]],['a']),([['a','a'],[]],['a'])]:
            with self.assertRaises(ValueError): selection_matrix(sets,features)

    def test_invariance_under_feature_and_run_order(self):
        sets = [['a','b'],['a'],['c'],['a','c']]
        a = stability_summary(sets,['a','b','c']); b = stability_summary(sets[::-1],['c','b','a'])
        self.assertAlmostEqual(a['nogueira'],b['nogueira'])
        self.assertAlmostEqual(a['jaccard_mean'],b['jaccard_mean'])

    def test_excluded_rows_are_physically_absent(self):
        toy_fixture.JointIntegrationTests.setUpClass(); original = toy_fixture.JointIntegrationTests.data
        changed = copy.deepcopy(original); positions = np.arange(60)
        changed.X.loc[60:,'x'] = 1e12; changed.y[60:] = 1-changed.y[60:]
        a=restrict_to_pool(original,positions); b=restrict_to_pool(changed,positions)
        self.assertTrue(a.X.equals(b.X)); np.testing.assert_array_equal(a.y,b.y)
        self.assertEqual(len(a.X),60); self.assertEqual(a.X.index.tolist(),list(range(60)))
        with self.assertRaises(ValueError): restrict_to_pool(original,[0,0])

    def test_durable_rejects_changed_completed_context(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'plan.json'; durable(path,{'x':[1,2]})
            self.assertEqual(durable(path,{'x':[1,2]}),{'x':[1,2]})
            with self.assertRaises(ValueError): durable(path,{'x':[2,1]})

    def test_complete_selection_and_verified_resume(self):
        fixture=toy_fixture.JointIntegrationTests; fixture.setUpClass()
        cfg={'replicates':2,'seed_root':20260929,'resample_rho':.8}
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
            root=Path(folder)
            result=run_replicate(root,fixture.data,cfg,fixture.config,fixture.base,0)
            self.assertEqual(result['counts']['selectors'],16)
            self.assertEqual(result['counts']['games'],32)
            self.assertEqual(result['counts']['final_classifier_fits'],0)
            self.assertEqual(len(result['selections']),8)
            self.assertFalse(result['outer_test_evaluated'])
            before={str(p):digest(p) for p in root.rglob('*.json')}
            self.assertEqual(result,run_replicate(root,fixture.data,cfg,fixture.config,fixture.base,0))
            self.assertEqual(before,{str(p):digest(p) for p in root.rglob('*.json')})
            target=root/'replicate-0/inner_splits.json'; target.write_text('{}','utf-8')
            with self.assertRaises(ValueError): run_replicate(root,fixture.data,cfg,fixture.config,fixture.base,0)


if __name__ == '__main__': unittest.main()
