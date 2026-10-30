"""No-real-fit candidate tests; stubs exercise orchestration and accounting."""
import unittest
import numpy as np
from a3.design import METHODS, resource_inventory, seed, seed_inventory
from a3.selection import (TrainingPool, FitOperations, SklearnBackend, ass_score_path,
    contribution_pool, final_evaluate, forward_select, l1_select, select_all, stable_top,
    stratified_folds, UtilityGame)
from a3.shapley_bound import corrected_scores, exact_shapley, permutation_shapley


class MemoryHook:
    enabled = True

    def __init__(self):
        self.events = []

    def reserve(self, kind, metadata):
        self.events.append({'kind': kind, 'metadata': metadata, 'status': 'started'})
        return len(self.events) - 1

    def finish(self, number, status, seconds, error=None):
        self.events[number].update(status=status, seconds=seconds, error=error)


class IdentityEncoder:
    def transform(self, X):
        return X


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, d):
        self.coef_ = np.zeros((1, d))
        self.feature_importances_ = np.zeros(d)

    def predict_proba(self, X):
        p = np.full(len(X), .6)
        return np.column_stack((1 - p, p))


class FakeBackend:
    is_test_double = True

    def __init__(self):
        self.encoders, self.models = [], []

    def fit_encoder(self, X):
        self.encoders.append(np.array(X))
        return IdentityEncoder()

    def fit_model(self, X, y, kind, model_seed, C=1.):
        self.models.append((np.array(X), np.array(y), kind, model_seed, C))
        return FakeModel(X.shape[1])


def pool():
    X = np.arange(480, dtype=float).reshape(60, 8)
    return TrainingPool(X, np.tile([0, 1], 30), tuple(f'row{i:03}' for i in range(60)),
                        tuple(f'x{i}' for i in range(8)), ('heldout',))


class A3NoFitTests(unittest.TestCase):
    def ops(self):
        self.backend, self.hook = FakeBackend(), MemoryHook()
        return FitOperations(self.backend, self.hook, 0)

    def test_seed_inventory_collision_and_scope(self):
        inv = seed_inventory()
        self.assertEqual(inv, seed_inventory())
        self.assertEqual(inv['count'], 307)
        self.assertEqual(len({r['seed'] for r in inv['records']}), inv['count'])
        with self.assertRaises(ValueError):
            seed(True, 'final_model')
        with self.assertRaises(ValueError):
            seed(0, 'unknown')
        with self.assertRaises(ValueError):
            seed(0, 'final_model', 1)
        with self.assertRaises(ValueError):
            seed(1, 'probe_lr', 0)

    def test_resource_arithmetic_includes_replay_audit_and_probe(self):
        inv = resource_inventory()
        self.assertEqual(inv['planned'], {'classifier': 19189, 'encoder': 19195})
        self.assertEqual(inv['failure_reserve'], {'classifier': 5811, 'encoder': 10805})
        self.assertEqual(sum(r['classifier'] for r in inv['rows'] if r['phase'] == 'main'), 15885)

    def test_pool_rejects_overlap_and_copies_inputs(self):
        p = pool()
        with self.assertRaises(ValueError):
            TrainingPool(p.X, p.y, p.row_ids, p.features, (p.row_ids[0],))
        with self.assertRaises(ValueError):
            TrainingPool(p.X, p.y, p.row_ids, p.features, role='test')
        self.assertFalse(p.X.flags.writeable)
        self.assertFalse(p.y.flags.writeable)

    def test_training_splits_and_subsamples_are_exclusive(self):
        p = pool()
        folds = stratified_folds(p, seed(0, 'inner_split'))
        self.assertEqual(sorted(np.concatenate([va for _, va in folds]).tolist()), list(range(60)))
        for tr, va in folds:
            self.assertFalse(set(tr) & set(va))
            self.assertEqual(set(p.y[va]), {0, 1})
        sub = contribution_pool(p, 0, 0)
        self.assertEqual(len(sub.X), 48)
        self.assertEqual(np.bincount(sub.y).tolist(), [24, 24])
        self.assertTrue(set(sub.row_ids) <= set(p.row_ids))
        self.assertEqual(sub.row_ids, contribution_pool(p, 0, 0).row_ids)

    def test_exact_guard_and_additive_math(self):
        utility = lambda S: sum([.2, -.1, .4][i] for i in S)
        exact = exact_shapley(utility, (0, 1, 2))
        estimated = permutation_shapley(utility, (0, 1, 2), 16, 71)
        np.testing.assert_allclose(exact, [.2, -.1, .4], atol=1e-14)
        np.testing.assert_allclose(exact, estimated['phi'], atol=1e-14)
        with self.assertRaises(ValueError):
            exact_shapley(lambda S: 0, tuple(range(9)))

    def test_corrected_variance_and_negative_forward_increments(self):
        result = corrected_scores([[1, 2], [1, 2]], [[.2, .3], [.2, .3]], .5)
        np.testing.assert_array_equal(result['corrected_sd'], [0, 0])
        self.assertTrue(result['clipped_to_zero'].all())
        path, increments = ass_score_path([-1.] * 8, np.zeros((8, 8)))
        self.assertEqual(path[:4], (0, 1, 2, 3))
        self.assertTrue(all(v < 0 for v in increments))
        chosen, _ = stable_top([0.] * 8)
        self.assertEqual(chosen, (0, 1, 2, 3))

    def test_real_fits_are_blocked_before_reservation(self):
        hook = MemoryHook()
        ops = FitOperations(SklearnBackend(), hook, 0)
        with self.assertRaisesRegex(RuntimeError, 'Real fits disabled'):
            ops.encoder(pool().X, METHODS[0], ['must_not_fit'])
        self.assertEqual(hook.events, [])

    def test_failure_is_one_counted_attempt_without_retry(self):
        ops = self.ops()
        def fail():
            raise RuntimeError('intentional stub failure')
        with self.assertRaisesRegex(RuntimeError, 'intentional stub failure'):
            ops.call('classifier', METHODS[2], 'l1', ['test'], fail, 20, 8)
        self.assertEqual(len(self.hook.events), 1)
        self.assertEqual(self.hook.events[0]['status'], 'failed')
        self.assertGreaterEqual(self.hook.events[0]['seconds'], 0)

    def test_l1_nested_refits_ranking_inside_each_training_fold(self):
        p, ops = pool(), self.ops()
        chosen, diag = l1_select(p, ops, stratified_folds(p, seed(0, 'inner_split')))
        self.assertEqual(chosen, (0, 1, 2, 3))
        self.assertEqual(diag['chosen_C'], .01)
        self.assertEqual(len(self.backend.models), 31)
        for X, y, _, _, _ in self.backend.models[:-1]:
            self.assertEqual(len(X), 40)
        self.assertEqual(len(self.backend.models[-1][0]), 60)
        self.assertEqual(diag['rebuild']['zero_count'], 8)

    def test_forward_negative_or_tied_scores_still_select_four(self):
        p, ops = pool(), self.ops()
        chosen, diag = forward_select(p, ops, stratified_folds(p, seed(0, 'inner_split')))
        self.assertEqual(chosen, (0, 1, 2, 3))
        self.assertEqual(len(self.backend.models), 78)
        self.assertFalse(diag['negative_gain_stops'])

    def test_utility_cache_and_constant_are_not_hidden_fits(self):
        p, ops = pool(), self.ops()
        game = UtilityGame(contribution_pool(p, 0, 0), ops, 0)
        game(())
        self.assertEqual(len(self.hook.events), 0)
        self.assertEqual(game.constant_evaluations, 3)
        v = game((0, 1))
        self.assertEqual(v, game((1, 0)))
        self.assertEqual(len(self.backend.models), 3)
        self.assertEqual(game.cache_hits, 1)

    def test_complete_seven_arm_stub_run_and_common_final_lr(self):
        p, ops = pool(), self.ops()
        result = select_all(p, ops)
        self.assertEqual(set(result['selected']), set(METHODS))
        self.assertTrue(all(len(result['selected'][m]) == 4 for m in METHODS[:-1]))
        self.assertEqual(result['diagnostics'][METHODS[1]]['alias'], '000')
        self.assertLessEqual(len(self.backend.models), 3170)
        self.assertEqual(len(self.backend.encoders), len(self.backend.models) + 1)
        self.assertTrue(all(e['metadata']['simulated'] for e in self.hook.events))
        self.assertTrue(all(e['status'] == 'succeeded' for e in self.hook.events))
        before = len(self.backend.models)
        final_ops = FitOperations(self.backend, self.hook, 0, phase='final')
        results = final_evaluate(p, p.X[:4] + 10000, [0, 1, 0, 1], ('a', 'b', 'c', 'd'), result['selected'], final_ops)
        final_models = self.backend.models[before:]
        self.assertEqual(len(final_models), 7)
        self.assertEqual({r[2] for r in final_models}, {'lr'})
        self.assertEqual({r[3] for r in final_models}, {seed(0, 'final_model')})
        self.assertTrue(all(len(r[0]) == 60 for r in final_models))
        self.assertEqual(results[METHODS[6]]['actual_k'], 8)
        with self.assertRaises(ValueError):
            final_evaluate(p, p.X[:4], [0, 1, 0, 1], p.row_ids[:4], result['selected'], final_ops)


if __name__ == '__main__':
    unittest.main()
