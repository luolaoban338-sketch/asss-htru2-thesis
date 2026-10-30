"""No-fit checks: synthetic generation, independent math identities, boundaries."""
import inspect
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from a4 import core


def toy_table(b=0):
    weights = np.array([.2, .1, -.05, .03, 0., .02]) + b * .01
    return np.array([sum(weights[j] for j in range(6) if mask & (1 << j))
                     + (.17 if mask & 3 == 3 else 0.)
                     - (.04 if mask & 28 == 28 else 0.) for mask in range(64)])


class A4NoFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blockers = [patch.object(k, "fit", side_effect=AssertionError("Real fit forbidden in A3.0"))
                        for k in (LogisticRegression, RandomForestClassifier, KMeans, StandardScaler)]
        for blocker in cls.blockers:
            blocker.start()

    @classmethod
    def tearDownClass(cls):
        for blocker in reversed(cls.blockers):
            blocker.stop()

    def test_all_seeds_declared_and_collision_free(self):
        manifest = core.seed_manifest()
        self.assertEqual(manifest["unique_coordinates"], 13336)
        self.assertEqual(manifest["records_sha256"], "eaf152a9de15f60e5c27851156b64ba48d71d05a5609d981bbd1e9f39c1ff2aa")
        self.assertNotEqual(core.coordinate_seed("a", 1), core.coordinate_seed("a", "1"))
        records = {tuple(r["coordinate"]): r["seed"] for r in core.seed_records()}
        self.assertEqual(len(records), len(set(records.values())))
        for b in range(4):
            self.assertIn(("signal_noise_LR", core.DATA_SEEDS[0], "main_permutation", b), records)
            self.assertIn(("signal_noise_LR", core.DATA_SEEDS[0], "utility_model", b, 0), records)

    def test_generation_reproducible_and_xor_shared(self):
        for scenario in core.SCENARIOS:
            one = core.generate(scenario, core.DATA_SEEDS[0])
            two = core.generate(scenario, core.DATA_SEEDS[0])
            self.assertEqual(one.train_X.shape, (600, 6))
            self.assertEqual(one.test_X.shape, (2000, 6))
            self.assertEqual(set(one.train_y), {0, 1})
            np.testing.assert_array_equal(one.train_X, two.train_X)
            np.testing.assert_array_equal(one.test_y, two.test_y)
            self.assertFalse(one.train_X.flags.writeable)
            self.assertFalse(np.array_equal(one.train_X, one.test_X[:600]))
        lr = core.generate_layer("xor_LR", core.DATA_SEEDS[0])
        rf = core.generate_layer("xor_RF", core.DATA_SEEDS[0])
        np.testing.assert_array_equal(lr.train_X, rf.train_X)
        np.testing.assert_array_equal(lr.test_y, rf.test_y)

    def test_full_nested_schedule_has_no_test_or_partition_overlap(self):
        # All 24 actual data pools, with all six/four/two declared coordinates.
        for scenario, root in itertools.product(core.SCENARIOS, core.DATA_SEEDS):
            data = core.generate(scenario, root)
            schedule = core.nested_training_schedule(data.train_y, scenario, root)
            self.assertEqual(len(schedule), 6)
            for selector in schedule:
                selected = set(selector["rows"])
                self.assertEqual(len(selected), 480)
                self.assertTrue(selected <= set(range(600)))
                self.assertEqual(len(selector["games"]), 4)
                for game in selector["games"]:
                    rows = set(game["rows"])
                    self.assertEqual(len(rows), 384)
                    self.assertTrue(rows <= selected)
                    validation_sets = []
                    for fold in game["folds"]:
                        tr, va = set(fold["train"]), set(fold["validation"])
                        self.assertFalse(tr & va)
                        self.assertEqual(tr | va, rows)
                        self.assertEqual(set(data.train_y[list(tr)]), {0, 1})
                        self.assertEqual(set(data.train_y[list(va)]), {0, 1})
                        validation_sets.append(va)
                    self.assertFalse(validation_sets[0] & validation_sets[1])
                    self.assertEqual(set.union(*validation_sets), rows)

    def test_exact_reference_matches_exhaustive_permutation_identity(self):
        table = toy_table()
        marginals = []
        for order in itertools.permutations(range(6)):
            mask = 0
            delta = np.zeros(6)
            for j in order:
                new = mask | (1 << j)
                delta[j] = table[new] - table[mask]
                mask = new
            marginals.append(delta)
        np.testing.assert_allclose(core.exact_shapley(table), np.mean(marginals, axis=0), atol=1e-12, rtol=0)
        expected = np.array([.2 + .17 / 2, .1 + .17 / 2, -.05 - .04 / 3, .03 - .04 / 3, -.04 / 3, .02])
        np.testing.assert_allclose(core.exact_shapley(table), expected, atol=1e-12, rtol=0)

    def test_exact_guard_and_invalid_tables(self):
        for d in (0, 9):
            with self.assertRaises(ValueError):
                core.exact_shapley(np.zeros(1 << max(0, d)), d)
        with self.assertRaises(ValueError):
            core.exact_shapley(np.zeros(63))
        with self.assertRaises(ValueError):
            core.exact_shapley([float("nan")] * 64)

    def test_M16_is_prefix_with_unbiased_MC_variance(self):
        a = core.permutation_estimate(toy_table(), 64, 91827)
        b = core.permutation_estimate(toy_table(), 16, 91827)
        np.testing.assert_array_equal(a["permutations"][:16], b["permutations"])
        np.testing.assert_array_equal(a["marginals"][:16], b["marginals"])
        np.testing.assert_allclose(b["mc_variance_of_mean"], b["marginals"].var(axis=0, ddof=1) / 16)

    def test_variance_formula_clipping_and_aliases(self):
        phi = np.array([[0., .1, .2, .3, .4, .5], [.1, .2, .3, .4, .5, .6],
                        [.2, .3, .4, .5, .6, .7], [.3, .4, .5, .6, .7, .8]])
        mc = np.full((4, 6), .04)
        stats = core.variance_components(phi, mc)
        np.testing.assert_allclose(stats["uncorrected"], np.full(6, 1 / 60))
        np.testing.assert_allclose(stats["raw_corrected"], np.full(6, 1 / 60 - .04))
        np.testing.assert_array_equal(stats["clipped_corrected"], np.zeros(6))
        labels = core.select_variants(core.MainPermutationEstimates(phi, mc), np.zeros((6, 6)))
        self.assertEqual(tuple(labels), core.CSR)
        for alias, canonical in core.ALIASES.items():
            self.assertEqual(labels[alias]["path"], labels[canonical]["path"])
            self.assertEqual(labels[alias]["structural_alias_of"], canonical)

    def test_selector_rejects_exact_or_diagnostic_interfaces(self):
        exact = np.array([core.exact_shapley(toy_table(b)) for b in range(4)])
        with self.assertRaises(TypeError):
            core.select_variants(exact, np.zeros((6, 6)))
        with self.assertRaises(TypeError):
            core.select_variants(core.MainPermutationEstimates(exact, np.zeros((4, 6)), 16), np.zeros((6, 6)))
        self.assertEqual(tuple(inspect.signature(core.select_variants).parameters), ("main", "redundancy"))
        self.assertEqual(tuple(inspect.signature(core.nested_training_schedule).parameters), ("train_y", "scenario", "data_seed"))

    def test_variance_diagnostics_target_is_finite_B_exact(self):
        tables = np.array([toy_table(b) for b in range(4)])
        result = core.variance_diagnostics(tables, "signal_noise", core.DATA_SEEDS[0], 0)
        np.testing.assert_allclose(result["target"], np.full(6, .01 ** 2 * 5 / 3), atol=1e-12)
        self.assertFalse(result["selection_eligible"])
        for M in ("16", "64"):
            self.assertEqual(np.shape(result["M"][M]["raw_corrected"]["values"]), (20, 6))
            self.assertGreaterEqual(result["M"][M]["clipping_rate"], 0)
            self.assertLessEqual(result["M"][M]["clipping_rate"], 1)

    def test_training_redundancy_and_deterministic_negative_path(self):
        X = np.column_stack([np.array([1, 1, 2, 3])] * 5 + [np.zeros(4)])
        R = core.redundancy_matrix(X)
        self.assertAlmostEqual(R[0, 1], 1.)
        self.assertEqual(R[0, 5], 0.)
        np.testing.assert_array_equal(np.diag(R), np.zeros(6))
        self.assertEqual(core.forward_path(-np.ones(6), np.zeros((6, 6)), .1), tuple(range(6)))
        scores = np.array([1, .99, .985, .1, 0., 0.])
        R = np.zeros((6, 6)); R[0, 1] = R[1, 0] = 1
        self.assertEqual(core.forward_path(scores, R, .1)[:2], (0, 2))

    def test_true_groups_stability_and_dataset_aggregation(self):
        self.assertEqual(core.recovery_metrics([0, 2], "correlated_groups")["group_coverage"], 1)
        redundant = core.recovery_metrics([0, 1], "correlated_groups")
        self.assertEqual(redundant["group_coverage"], .5)
        self.assertEqual(redundant["same_group_excess"], 1)
        self.assertEqual(core.recovery_metrics([0, 1], "xor")["joint_signal_recovery"], 1)
        same = core.stability_metrics([[0, 1]] * 6)
        self.assertEqual(same["nogueira"], 1.)
        self.assertEqual(same["mean_jaccard"], 1.)
        self.assertEqual(len(same["pairwise_jaccard"]), 15)
        self.assertIsNone(core.stability_metrics([list(range(6))] * 6)["nogueira"])
        summary = core.paired_dataset_summary([-2, -1, 0, 0, 0, 1, 2, 3])
        self.assertEqual((summary["negative"], summary["zero"], summary["positive"]), (2, 3, 3))
        with self.assertRaises(ValueError):
            core.paired_dataset_summary([0] * 20)

    def test_exact_resource_reconciliation_and_timing_stop(self):
        inventory = core.resource_inventory()
        self.assertEqual(inventory["planned_classifier_attempt_bound"], 105648)
        self.assertEqual(inventory["planned_encoder_attempt_bound"], 105648)
        self.assertEqual(inventory["uncommitted_attempt_reserve"], 6352)
        self.assertEqual(inventory["LR_classifier_attempt_bound"] + inventory["RF_classifier_attempt_bound"], 105648)
        targets = core.audit_targets()
        self.assertEqual(len(targets), 7)
        self.assertEqual([t["mask"] for t in targets[:6]], [1, 3, 7, 15, 31, 63])
        self.assertEqual(targets[-1], {"kind": "final", "csr": "111", "k": 2})
        self.assertEqual(core.timing_projection([.01] * 16, [.01] * 16)["status"], "pass")
        self.assertEqual(core.timing_projection([1.] * 16, [1.] * 16)["status"], "stop_defer")

    def test_protocol_matches_implemented_constants(self):
        path = Path(__file__).resolve().parents[1] / "protocols" / "a4.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["seed_manifest"], core.seed_manifest())
        self.assertEqual(protocol["resource_inventory"], core.resource_inventory())
        self.assertEqual(protocol["direct_audit_targets_per_selector"], core.audit_targets())
        self.assertEqual(protocol["status"], "scientific_design_and_no_fit_primitives_frozen_execution_pending")
        for path in protocol["storage"].values():
            self.assertTrue(path.startswith("F:\\ChatGPT\\论文大数据\\"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
