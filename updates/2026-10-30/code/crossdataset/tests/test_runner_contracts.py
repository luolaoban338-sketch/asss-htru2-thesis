"""Runner authorization, durable decisions and unchanged-resume contracts.

Every learner/encoder/cluster fit is forbidden in these unit tests. Generated
artifacts live only in disposable test directories.
"""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data import Dataset
from src.preprocessing import TabularEncoder
from experiments.joint_selection import candidate_grid
from scripts import freeze_execution as freeze
from scripts import run_stage as runner
from scripts.selection_pipeline import cached, commit, read, save


class RunnerContractsTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.manifest = self.root / 'protocols' / 'freeze_manifest.json'
        save(self.manifest, {'unit_test_fixture': True})
        self.stack.enter_context(patch.object(runner, 'ROOT', self.root))
        self.stack.enter_context(patch.object(runner, 'MANIFEST', self.manifest))
        self.verify = self.stack.enter_context(patch.object(runner, 'verify', return_value={'verified': True}))
        for model in (LogisticRegression, RandomForestClassifier, KMeans, TabularEncoder):
            self.stack.enter_context(patch.object(model, 'fit', side_effect=AssertionError('No fits allowed in runner contract tests')))
        self.a1 = {'packages': ['wdbc-lr'], 'purpose': 'unit-test-only'}
        self.a2 = {'datasets': ['wdbc', 'adult_train'], 'outer_folds': [0], 'inner_folds': 3,
                   'seed_root': 17, 'methods': ['full', 'random_half', 'ordinary_half', 'robust_half'],
                   'lambda_grid': [0, .5], 'eta_grid': [0, .1], 'epsilon': .01,
                   'numeric_tolerance': 1e-12, 'limits': {'wdbc': {}},
                   'comparisons': ['full', 'sample_only', 'feature_only', 'joint', 'sequential',
                                   'joint_random', 'joint_ordinary', 'joint_robust']}

    def config_read(self, path):
        path = Path(path)
        if path.name == 'preflight.json':
            return self.a1
        if path.name == 'comparison.json':
            return self.a2
        return read(path)

    def complete_package(self):
        dest = self.root / 'runs' / runner.A1_RUN / 'wdbc-lr'
        commit(dest / 'proof.json', {'completed': True})
        commit(dest / 'package.json', {'status': 'passed', 'artifact_sha256': runner.complete_hashes(dest)})
        return dest

    def test_completed_preflight_checks_inputs_then_skips_before_budget_open(self):
        self.complete_package()
        with patch.object(runner, 'read', side_effect=self.config_read), \
             patch.object(runner, 'inputs', return_value=(None, None, None)), \
             patch.object(runner, 'Budget', side_effect=AssertionError('Completed package opened ledger')) as budget:
            runner.preflight('wdbc-lr')
        self.verify.assert_called_once_with()
        budget.assert_not_called()
        self.assertTrue(cached(self.root / 'runs' / runner.A1_RUN / 'manifest.json'))
        self.assertFalse((self.root / 'runs' / runner.A1_RUN / 'fit_budget.sqlite3').exists())

    def test_completed_preflight_still_rejects_frozen_input_drift(self):
        self.complete_package()
        run = self.root / 'runs' / runner.A1_RUN
        runner.prepare(run, {'different_previous_configuration': True})
        with patch.object(runner, 'read', side_effect=self.config_read), \
             patch.object(runner, 'inputs', return_value=(None, None, None)), \
             patch.object(runner, 'Budget', side_effect=AssertionError('Drift opened ledger')):
            with self.assertRaisesRegex(RuntimeError, 'Run input changed'):
                runner.preflight('wdbc-lr')

    def test_completed_package_rejects_content_tamper_and_extra_artifact(self):
        dest = self.complete_package()
        self.assertTrue(runner.completed(dest))
        save(dest / 'extra.json', {'unexpected': True})
        with self.assertRaisesRegex(RuntimeError, 'Completed package changed'):
            runner.completed(dest)
        (dest / 'extra.json').unlink()
        save(dest / 'proof.json', {'completed': False})
        with self.assertRaisesRegex(RuntimeError, 'Completed package changed'):
            runner.completed(dest)

    def test_completed_package_marker_rejects_manifest_tamper(self):
        dest = self.complete_package()
        save(dest / 'package.json', {'status': 'tampered', 'artifact_sha256': {}})
        with self.assertRaisesRegex(ValueError, 'Completed artifact modified'):
            runner.completed(dest)

    def test_comparison_requires_authorization_before_data_or_budget(self):
        with patch.object(runner, 'read', side_effect=self.config_read), \
             patch.object(runner, 'inputs', side_effect=AssertionError('Unauthorized data access')) as inputs, \
             patch.object(runner, 'Budget', side_effect=AssertionError('Unauthorized budget access')) as budget:
            with self.assertRaisesRegex(ValueError, 'execute-authorized-package'):
                runner.compare('wdbc', 0, False)
        inputs.assert_not_called()
        budget.assert_not_called()

    def test_comparison_cli_authorization_defaults_false_and_requires_flag(self):
        arguments = ['run_stage.py', 'compare', '--dataset', 'wdbc', '--fold', '0']
        with patch('sys.argv', arguments), patch.object(runner, 'compare') as compare:
            runner.main()
            compare.assert_called_once_with('wdbc', 0, False)
        with patch('sys.argv', arguments + ['--execute-authorized-package']), patch.object(runner, 'compare') as compare:
            runner.main()
            compare.assert_called_once_with('wdbc', 0, True)

    def test_missing_adult_runtime_gate_fails_before_data_or_budget(self):
        with patch.object(runner, 'read', side_effect=self.config_read), \
             patch.object(runner, 'verify_preflight_gate', return_value={'status': 'passed'}), \
             patch.object(runner, 'inputs', side_effect=AssertionError('Adult data loaded before gate')) as inputs, \
             patch.object(runner, 'Budget', side_effect=AssertionError('Adult ledger opened before gate')) as budget:
            with self.assertRaisesRegex(RuntimeError, 'runtime feasibility gate'):
                runner.compare('adult_train', 0, True)
        inputs.assert_not_called()
        budget.assert_not_called()

    def test_missing_preflight_audit_gate_fails_before_data_or_budget(self):
        with patch.object(runner, 'read', side_effect=self.config_read), \
             patch.object(runner, 'inputs', side_effect=AssertionError('A2 data loaded before A1 gate')) as inputs, \
             patch.object(runner, 'Budget', side_effect=AssertionError('A2 ledger opened before A1 gate')) as budget:
            with self.assertRaisesRegex(RuntimeError, 'A1 independent audit must pass'):
                runner.compare('wdbc', 0, True)
        inputs.assert_not_called()
        budget.assert_not_called()

    def test_outer_decisions_are_durable_before_refit_and_evaluation(self):
        n = 12
        data = Dataset('wdbc', pd.DataFrame({'a': np.arange(n), 'b': np.arange(n) / 2}),
            np.tile([0, 1], n // 2), np.array([f'r{i}' for i in range(n)]),
            np.array([f'g{i}' for i in range(n)]), pd.DataFrame(index=range(n)), ('a', 'b'), (), 'toy-sha')
        splits = {'outer_folds': [{'train': {'positions': list(range(8))}, 'test': {'positions': list(range(8, 12))},
            'inner_folds': [{'fold': i, 'train': {'positions': [0, 1, 2, 3]},
                             'validation': {'positions': [4, 5, 6, 7]}} for i in range(3)]}]}
        base = {'model': {'class': 'LogisticRegression'}}
        dest = self.root / 'runs' / 'comparison-wdbc-lr-20261014-v1' / 'fold-0'
        decision_path = dest / 'decisions.json'
        events = []

        def fake_selectors(data, train, partition_id, config, base, directory):
            if str(partition_id).endswith(':refit'):
                persisted = cached(decision_path)
                self.assertIsNotNone(persisted)
                self.assertEqual(len(persisted['candidate_scores']), len(candidate_grid(config, 2)))
                events.append('refit_after_durable_decisions')
            return {name: {'features': ['a', 'b'], 'selected_positions': train.tolist(), 'actual_rho': 1,
                'paths': [{'lam': lam, 'eta': eta, 'path': [{'features': ['a']}, {'features': ['a', 'b']}]}
                          for lam in config['lambda_grid'] for eta in config['eta_grid']]}
                    for name in config['methods']}

        def fake_candidates(data, train, validation, selectors, config, base, seed, directory):
            events.append('inner_candidates')
            return [{**c, 'f1': .9, 'actual_rho': 1} for c in candidate_grid(config, 2)]

        def fake_evaluate(data, train, selected, validation, features, base, seed, path, *, config):
            self.assertIsNotNone(cached(decision_path))
            self.assertEqual(set(validation), {8, 9, 10, 11})
            events.append('outer_evaluate_after_durable_decisions')
            return commit(path, {'metrics': {'f1': .9}, 'unit_test_fake': True})

        budget = MagicMock()
        budget.instrument.side_effect = lambda: nullcontext()
        budget.track.side_effect = lambda tag: nullcontext()
        budget.counts.return_value = []
        with patch.object(runner, 'read', side_effect=self.config_read), \
             patch.object(runner, 'verify_preflight_gate', return_value={'status': 'passed'}), \
             patch.object(runner, 'inputs', return_value=(data, splits, base)), \
             patch.object(runner, 'Budget', return_value=budget), \
             patch.object(runner, 'threadpool_limits', side_effect=lambda **kwargs: nullcontext()), \
             patch.object(runner, 'selectors', side_effect=fake_selectors), \
             patch.object(runner, 'evaluate_candidates', side_effect=fake_candidates), \
             patch.object(runner, 'fit_evaluate', side_effect=fake_evaluate):
            runner.compare('wdbc', 0, True)
        self.assertEqual(events[:3], ['inner_candidates'] * 3)
        self.assertEqual(events[3], 'refit_after_durable_decisions')
        self.assertEqual(events[4:], ['outer_evaluate_after_durable_decisions'] * 8)
        self.assertTrue(runner.completed(dest))
        budget.close.assert_called_once_with()

    def test_frozen_manifest_detects_environment_and_file_drift(self):
        save(self.manifest, {'environment': {'python': 'frozen'}, 'files': {'src/model.py': 'old-sha'}})
        with patch.object(freeze, 'MANIFEST', self.manifest), \
             patch.object(freeze, 'environment', return_value={'python': 'frozen'}), \
             patch.object(freeze, 'files', return_value={'src/model.py': 'old-sha'}):
            self.assertEqual(freeze.verify()['environment'], {'python': 'frozen'})
        with patch.object(freeze, 'MANIFEST', self.manifest), \
             patch.object(freeze, 'environment', return_value={'python': 'changed'}):
            with self.assertRaisesRegex(RuntimeError, 'environment changed'):
                freeze.verify()
        with patch.object(freeze, 'MANIFEST', self.manifest), \
             patch.object(freeze, 'environment', return_value={'python': 'frozen'}), \
             patch.object(freeze, 'files', return_value={'src/model.py': 'changed-sha'}):
            with self.assertRaisesRegex(RuntimeError, 'code/config/data/partitions changed'):
                freeze.verify()


if __name__ == '__main__':
    unittest.main()
