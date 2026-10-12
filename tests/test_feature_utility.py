import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
from src.contracts import TrainingPartition
from src.preprocessing import TabularEncoder
from src.evaluation import classification_metrics
from experiments.feature_utility import FeatureUtility
from experiments.shapley import exact_shapley, permutation_shapley, corrected_scores


class ShapleyTests(unittest.TestCase):
    def test_exact_additive_interaction_dummy_and_nonzero_empty(self):
        def game(S):
            return .2 + .1 * ('a' in S) + .3 * ('b' in S) + .2 * ({'a', 'b'} <= set(S))
        exact = exact_shapley(game, ['a', 'b', 'dummy'])
        np.testing.assert_allclose(exact, [.2, .4, 0], atol=1e-15)
        estimate = permutation_shapley(game, ['a', 'b', 'dummy'], 16, 17)
        np.testing.assert_allclose(estimate['marginals'].sum(axis=1), .6, atol=1e-15)
        self.assertTrue(np.all(estimate['marginals'][:, 2] == 0))

    def test_negative_contributions_preserved(self):
        result = exact_shapley(lambda S: .5 - .2 * ('bad' in S), ['bad', 'dummy'])
        np.testing.assert_allclose(result, [-.2, 0], atol=1e-15)

    def test_additive_permutations_are_exact_with_zero_error(self):
        result = permutation_shapley(lambda S: .25 + sum({'a': .125, 'b': .25}[s] for s in S), ['a', 'b'], 16, 1)
        np.testing.assert_array_equal(result['phi'], [.125, .25])
        np.testing.assert_array_equal(result['mc_variance_of_mean'], [0, 0])

    def test_mc_variance_is_variance_of_mean(self):
        game = lambda S: float({'a', 'b'} <= set(S))
        a = permutation_shapley(game, ['a', 'b'], 16, 4)
        b = permutation_shapley(game, ['a', 'b'], 16, 4)
        np.testing.assert_array_equal(a['permutations'], b['permutations'])
        np.testing.assert_allclose(a['mc_variance_of_mean'], a['marginals'].var(axis=0, ddof=1) / 16)

    def test_variance_correction_clips_without_claiming_certainty(self):
        result = corrected_scores([[0, 0], [2, 1]], [[1, 1], [1, 1]], .5)
        np.testing.assert_allclose(result['observed_variance'], [2, .5])
        np.testing.assert_allclose(result['corrected_sd'], [1, 0])
        np.testing.assert_allclose(result['score'], [.5, .5])
        np.testing.assert_array_equal(result['clipped_to_zero'], [False, True])

    def test_invalid_budget_and_input(self):
        with self.assertRaises(ValueError): exact_shapley(lambda S: 0, list(range(9)))
        with self.assertRaises(ValueError): permutation_shapley(lambda S: 0, ['a'], 1, 0)
        with self.assertRaises(ValueError): corrected_scores([[1]], [[0]], .5)
        with self.assertRaises(ValueError): corrected_scores([[1], [2]], [[-1], [0]], .5)


class UtilityTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((Path(__file__).resolve().parents[1] / 'configs/baseline_pilot_v1.json').read_text('utf-8'))
        rng = np.random.default_rng(11)
        self.X = pd.DataFrame({'x': rng.normal(size=60), 'category': ['a', 'b', 'c'] * 20})
        self.part = TrainingPartition(tuple(f'r{i}' for i in range(60)), (0,) * 20 + (1,) * 40,
            tuple(self.X.columns), tuple(f'g{i // 2}' for i in range(60)), 'train', 'toy')
        self.method = self.base['methods'][1]
        self.limit = threadpool_limits(limits=1); self.limit.__enter__()
    def tearDown(self): self.limit.__exit__(None, None, None)
    def game(self): return FeatureUtility(self.X, self.part, ['x'], ['category'], self.method, self.base, 7)

    def test_selected_rows_fixed_and_groups_disjoint(self):
        game = self.game(); before = game.fold_metadata()
        for subset in [(), ('x',), ('category',), ('x', 'category')]: game.utility(subset)
        self.assertEqual(before, game.fold_metadata())
        for f in game.folds:
            self.assertFalse(set(game.ids[f['selected']]) & set(game.ids[f['validation']]))
            self.assertFalse(set(game.groups[f['train']]) & set(game.groups[f['validation']]))
            self.assertTrue(set(f['selected']) <= set(f['train']))
            self.assertLess(len(f['selected']), len(f['train']))

    def test_empty_is_selected_label_constant(self):
        game = self.game()
        scores = [classification_metrics(game.y[f['validation']], np.repeat(game.y[f['selected']].mean(), len(f['validation'])))['f1'] for f in game.folds]
        self.assertEqual(game.utility(()), np.mean(scores)); self.assertGreater(game.utility(()), 0)
        self.assertEqual(game.fit_count, 0); self.assertEqual(game.constant_count, 3)

    def test_full_utility_matches_direct_selected_row_models(self):
        game = self.game(); scores = []
        for f in game.folds:
            chosen, va = f['selected'], f['validation']
            enc = TabularEncoder(['x'], ['category'])
            Z = enc.fit_transform(self.X.iloc[chosen])
            params = {k: v for k, v in self.base['model'].items() if k != 'class'}
            model = LogisticRegression(**params, random_state=f['model_seed']).fit(Z, game.y[chosen])
            scores.append(classification_metrics(game.y[va], model.predict_proba(enc.transform(self.X.iloc[va]))[:, 1])['f1'])
        self.assertEqual(game.utility(game.features), np.mean(scores))

    def test_canonical_subset_cache_and_original_variable_blocks(self):
        game = self.game(); a = game.utility(['x', 'category']); fits = game.fit_count
        self.assertEqual(a, game.utility(['category', 'x'])); self.assertEqual(fits, game.fit_count)
        with self.assertRaises(ValueError): game.utility(['category=a'])
        with self.assertRaises(ValueError): game.utility(['x', 'x'])

    def test_cache_context_changes_with_values_method_seed(self):
        a = self.game(); self.X.loc[0, 'x'] += 1; b = self.game()
        self.assertNotEqual(a.context_hash, b.context_hash)
        c = FeatureUtility(self.X, self.part, ['x'], ['category'], self.base['methods'][0], self.base, 7)
        d = FeatureUtility(self.X, self.part, ['x'], ['category'], self.method, self.base, 8)
        self.assertNotEqual(b.context_hash, c.context_hash); self.assertNotEqual(b.context_hash, d.context_hash)

    def test_caller_mutation_does_not_change_game(self):
        game = self.game(); before = game.X.copy(); params = json.dumps(game.baseline, sort_keys=True)
        self.X.loc[0, 'x'] = 10000; self.base['model']['C'] = 100
        pd.testing.assert_frame_equal(game.X, before); self.assertEqual(json.dumps(game.baseline, sort_keys=True), params)

    def test_test_partition_rejected(self):
        p = TrainingPartition(self.part.row_ids, self.part.labels, self.part.feature_names, self.part.group_ids, 'test', 'bad')
        with self.assertRaises(ValueError): FeatureUtility(self.X, p, ['x'], ['category'], self.method, self.base, 7)


if __name__ == '__main__': unittest.main()
