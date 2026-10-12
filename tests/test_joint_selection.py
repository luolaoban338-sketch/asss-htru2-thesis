import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import Dataset
from experiments.joint_selection import choose_candidate, make_decisions, candidate_grid, aggregate_candidates, subset_for
from scripts.run_joint_pilot import build_selector, fit_evaluate, cached, commit

ROOT = Path(__file__).resolve().parents[1]


class JointDecisionTests(unittest.TestCase):
    def test_feasibility_and_lexicographic_order(self):
        rows = [dict(candidate_id=i, mean_f1=f, cell_fraction=c, k=k)
                for i, (f, c, k) in enumerate([(.79,.1,1), (.8,.2,2), (.81,.2,1), (.82,.2,1), (.82,.2,1)])]
        self.assertEqual(choose_candidate(rows, .8)['candidate_id'], 3)
        with self.assertRaises(ValueError): choose_candidate(rows, .9)

    def test_rounding_tolerance_does_not_accumulate(self):
        rows = [dict(candidate_id=i, mean_f1=.8, cell_fraction=c, k=1)
                for i, c in enumerate([.2 + 1.5e-12, .2 + .7e-12, .2])]
        self.assertEqual(choose_candidate(rows, .8)['candidate_id'], 1)

    def test_joint_and_sequential_share_original_threshold(self):
        cfg = json.loads((ROOT / 'configs/joint_pilot_v1.json').read_text('utf-8'))
        rows = []
        for c in candidate_grid(cfg, 2):
            rho = 1 if c['method'] == 'full' else .5
            f1 = .8 if c['k'] == 2 else (.781 if c['method'] != 'full' else .795)
            rows.append({**c, 'mean_f1': f1, 'mean_actual_rho': rho, 'cell_fraction': rho*c['k']/2})
        result = make_decisions(rows, cfg, 2)
        self.assertAlmostEqual(result['threshold'], .79)
        self.assertEqual(result['selections']['sample_only']['method'], 'random_half')
        self.assertEqual(result['selections']['sequential']['k'], 2)
        # Equal matrix fraction: feature-only k=1 wins against sample-only k=2.
        self.assertEqual(result['selections']['joint']['method'], 'full')
        self.assertEqual(result['selections']['joint']['k'], 1)

    def test_aggregation_rejects_missing_or_changed_grid(self):
        cfg = json.loads((ROOT / 'configs/joint_pilot_v1.json').read_text('utf-8'))
        rows = [{**c, 'f1': .7, 'actual_rho': 1} for c in candidate_grid(cfg, 2)]
        with self.assertRaises(ValueError): aggregate_candidates([rows, rows], cfg, 2)
        with self.assertRaises(ValueError): aggregate_candidates([rows, rows, rows[:-1]], cfg, 2)
        self.assertEqual(len(aggregate_candidates([rows]*3, cfg, 2)), 32)


class JointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((ROOT/'configs/baseline_pilot_v1.json').read_text('utf-8'))
        cls.config = json.loads((ROOT/'configs/joint_pilot_v1.json').read_text('utf-8'))
        cls.config.update(B=2, M=4, utility_folds=2)
        rng = np.random.default_rng(81)
        n = 80
        cls.data = Dataset('toy', pd.DataFrame({'x': rng.normal(size=n), 'cat': np.tile(['a','b','c','d'],20)}),
            np.tile([0,1],40), np.array([f'r{i}' for i in range(n)]), np.array([f'g{i}' for i in range(n)]),
            pd.DataFrame(index=range(n)), ('x',), ('cat',), 'toy')

    def test_heldout_perturbation_cannot_change_selector(self):
        train = np.arange(60); heldout = np.arange(60,80)
        changed = copy.deepcopy(self.data)
        changed.X.loc[heldout, 'x'] = 1e8; changed.X.loc[heldout, 'cat'] = 'unseen'
        changed.y[heldout] = 1 - changed.y[heldout]
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
            root = Path(folder); method = self.base['methods'][1]
            a = build_selector(self.data, train, 'toy_train', method, self.config, self.base, root/'a')
            b = build_selector(changed, train, 'toy_train', method, self.config, self.base, root/'b')
            self.assertEqual(a['selected_ids'], b['selected_ids'])
            self.assertEqual(a['redundancy'], b['redundancy']); self.assertEqual(a['paths'], b['paths'])
            candidate = {'lam': .5, 'eta': .1, 'k': 2}
            features = subset_for(a, candidate)
            self.assertEqual(features, ('x','cat'))
            fit = fit_evaluate(self.data, train, a['selected_positions'], heldout, features, self.base, 123, root/'model.json')
            self.assertEqual(fit['encoder']['fit_n_rows'], 30)
            self.assertEqual(fit['context']['features'], ['x','cat'])
            self.assertEqual(len(fit['probabilities']), 20)
            with self.assertRaises(ValueError):
                fit_evaluate(self.data, train, a['selected_positions'], np.array([0,61]), features, self.base, 123, root/'bad.json')

    def test_completed_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.json'
            commit(path, {'value': 1}); self.assertEqual(cached(path), {'value': 1})
            path.write_text('{"value":2}', 'utf-8')
            with self.assertRaises(ValueError): cached(path)


if __name__ == '__main__': unittest.main()
