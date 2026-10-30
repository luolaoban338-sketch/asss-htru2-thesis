"""A1 regression contracts. Every classifier fit is mocked in this test suite."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data import Dataset
from src.models import make_classifier, learner_metadata, fit_count_metadata
from experiments.feature_utility import FeatureUtility
from scripts import selection_pipeline as pipeline


def fake_fit(model, X, y):
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = X.shape[1]
    if isinstance(model, LogisticRegression):
        model.coef_ = np.zeros((1, X.shape[1]))
        model.intercept_ = np.zeros(1)
        model.n_iter_ = np.array([1])
    return model


def fake_predict(model, X):
    p = np.full(len(X), .6)
    return np.column_stack((1 - p, p))


@contextmanager
def fake_models():
    with patch.object(LogisticRegression, 'fit', fake_fit), \
         patch.object(LogisticRegression, 'predict_proba', fake_predict), \
         patch.object(RandomForestClassifier, 'fit', fake_fit), \
         patch.object(RandomForestClassifier, 'predict_proba', fake_predict):
        yield


class ModelsDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing local fake must fail instead of executing an unledgered fit.
        cls.fit_guards = [patch.object(learner, 'fit', side_effect=AssertionError('Real fit forbidden in A1 unit tests'))
                          for learner in (LogisticRegression, RandomForestClassifier)]
        for guard in cls.fit_guards:
            guard.start()

    @classmethod
    def tearDownClass(cls):
        for guard in reversed(cls.fit_guards):
            guard.stop()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dest = Path(self.temporary.name)
        self.stop_patch = patch.object(pipeline, 'ROOT', self.dest)
        self.utility_stop_patch = patch('experiments.feature_utility.ROOT', self.dest)
        self.stop_patch.start()
        self.utility_stop_patch.start()
        self.addCleanup(self.stop_patch.stop)
        self.addCleanup(self.utility_stop_patch.stop)
        n = 48
        frame = pd.DataFrame({'a': np.arange(n, dtype=float), 'b': np.sin(np.arange(n))})
        self.data = Dataset('toy', frame, np.tile([0, 1], n // 2),
            np.array([f'r{i}' for i in range(n)]), np.array([f'g{i}' for i in range(n)]),
            pd.DataFrame(index=range(n)), ('a', 'b'), (), 'toy-data-sha')
        self.method = {'name': 'full', 'rho': 1, 'strategy': 'stratified_random'}
        self.base = {'model': {'class': 'LogisticRegression', 'max_iter': 100},
                     'sampler': {}, 'threshold': .5, 'methods': [self.method]}
        self.config = {'seed_root': 17, 'B': 2, 'M': 2, 'subsample_rho': 1,
                       'utility_folds': 2, 'methods': ['full'], 'lambda_grid': [0],
                       'eta_grid': [0], 'exact_diagnostic': False, 'exact_max_features': 8}
        self.train = np.arange(36)
        self.validation = np.arange(36, 48)

    def selector(self):
        return pipeline.build_selector(self.data, self.train, 'toy:train', self.method,
                                       self.config, self.base, self.dest / 'selector')

    def fitted(self):
        return pipeline.fit_evaluate(self.data, self.train, self.train, self.validation,
            ('a', 'b'), self.base, 22, self.dest / 'model.json', config=self.config)

    def test_seeded_factories_and_truthful_counters(self):
        lr = make_classifier({'class': 'LogisticRegression', 'random_state': 9}, 22)
        rf = make_classifier({'class': 'RandomForestClassifier', 'n_estimators': 3}, 22)
        self.assertIsInstance(lr, LogisticRegression)
        self.assertIsInstance(rf, RandomForestClassifier)
        self.assertEqual(lr.random_state, 22)
        self.assertEqual(rf.random_state, 22)
        self.assertEqual(rf.n_jobs, 1)
        self.assertEqual(fit_count_metadata({'class': 'LogisticRegression'}, 3)['logistic_fits'], 3)
        self.assertEqual(fit_count_metadata({'class': 'RandomForestClassifier'}, 3),
                         {'model_class': 'RandomForestClassifier', 'classifier_fits': 3})
        for bad in [None, True, -1, 2 ** 32]:
            with self.assertRaises(ValueError):
                make_classifier({'class': 'LogisticRegression'}, bad)
        with self.assertRaises(ValueError):
            make_classifier({'class': 'RandomForestClassifier', 'n_jobs': 2}, 22)
        with self.assertRaises(ValueError):
            make_classifier({'class': 'SVC'}, 22)

    def test_metadata_does_not_require_linear_coefficients(self):
        lr = fake_fit(make_classifier(self.base['model'], 22), np.zeros((4, 2)), [0, 1, 0, 1])
        rf = fake_fit(make_classifier({'class': 'RandomForestClassifier'}, 22), np.zeros((4, 2)), [0, 1, 0, 1])
        self.assertIn('coef', learner_metadata(lr))
        self.assertEqual(learner_metadata(lr)['model_class'], 'LogisticRegression')
        self.assertNotIn('coef', learner_metadata(rf))
        self.assertEqual(learner_metadata(rf)['model_class'], 'RandomForestClassifier')

    def test_high_dimension_exact_never_calls_utility_or_enumerator(self):
        def forbidden(_):
            raise AssertionError('Exact utility evaluated on high-dimensional data')
        with patch.object(pipeline, 'exact_shapley', side_effect=AssertionError('Exact enumerator entered')):
            result = pipeline.exact_diagnostic(forbidden, tuple(f'x{i}' for i in range(30)),
                np.zeros(30), {'exact_diagnostic': True, 'exact_max_features': 8})
        self.assertEqual(result['status'], 'not_applicable')
        self.assertEqual(result['reason'], 'dimension_exceeds_limit')
        self.assertIsNone(result['exact_phi'])
        self.assertIsNone(result['max_abs_error'])

    def test_exact_disabled_and_limit_validation(self):
        def forbidden(_):
            raise AssertionError('Disabled exact utility evaluated')
        result = pipeline.exact_diagnostic(forbidden, ('a', 'b'), [0, 0], {'exact_diagnostic': False})
        self.assertEqual(result['status'], 'disabled')
        for config in [{'exact_diagnostic': 1}, {'exact_max_features': 9}, {'exact_max_features': True}]:
            with self.assertRaises(ValueError):
                pipeline.exact_diagnostic(forbidden, ('a', 'b'), [0, 0], config)

    def test_low_dimension_exact_still_computes(self):
        weights = {'a': .25, 'b': .5}
        result = pipeline.exact_diagnostic(lambda subset: sum(weights[x] for x in subset),
            ('a', 'b'), [.25, .5], {'exact_diagnostic': True})
        self.assertEqual(result['status'], 'completed')
        np.testing.assert_allclose(result['exact_phi'], [.25, .5])
        self.assertEqual(result['max_abs_error'], 0)

    def test_feature_utility_rf_and_cache_model_identity(self):
        lr_game = FeatureUtility(self.data.X, self.data.partition(np.arange(48), 'toy'),
            self.data.numeric, (), self.method, self.base, 22, 2)
        forest_base = deepcopy(self.base)
        forest_base['model'] = {'class': 'RandomForestClassifier', 'n_estimators': 3}
        rf_game = FeatureUtility(self.data.X, self.data.partition(np.arange(48), 'toy'),
            self.data.numeric, (), self.method, forest_base, 22, 2)
        self.assertNotEqual(lr_game.context_hash, rf_game.context_hash)
        with fake_models():
            value = rf_game.utility(('a',))
            self.assertEqual(value, rf_game.utility(('a',)))
        self.assertEqual(rf_game.fit_count, 2)
        self.assertEqual(rf_game.fit_metadata()['classifier_fits'], 2)
        self.assertNotIn('logistic_fits', rf_game.fit_metadata())
        metadata = rf_game.accounting_metadata()
        self.assertEqual(metadata['utility_requests'], 2)
        self.assertEqual(metadata['cache_hits'], 1)
        self.assertEqual(metadata['nonempty_unique_coalitions'], 1)
        self.assertTrue(metadata['fit_reconciliation_passed'])
        self.assertTrue(metadata['request_reconciliation_passed'])

    def test_model_cache_rejects_config_model_and_data_changes(self):
        with fake_models():
            first = self.fitted()
        self.assertEqual(self.fitted(), first)
        self.config['M'] += 1
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.fitted()
        self.config['M'] -= 1
        self.base['model'] = {'class': 'RandomForestClassifier', 'n_estimators': 3}
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.fitted()
        self.base['model'] = {'class': 'LogisticRegression', 'max_iter': 100}
        self.data.X.loc[0, 'a'] += 1
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.fitted()

    def test_selector_cache_rejects_configuration_changes(self):
        with fake_models():
            first = self.selector()
        self.assertEqual(first, self.selector())
        self.config['M'] += 1
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.selector()

    def test_partial_game_cache_rejects_model_change(self):
        with fake_models():
            self.selector()
        (self.dest / 'selector' / 'selector.json').unlink()
        (self.dest / 'selector' / 'selector.complete.json').unlink()
        self.base['model'] = {'class': 'RandomForestClassifier', 'n_estimators': 3}
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.selector()

    def test_rf_pipeline_metadata_has_no_logistic_counts(self):
        self.base['model'] = {'class': 'RandomForestClassifier', 'n_estimators': 3}
        with fake_models():
            selector = self.selector()
            model = self.fitted()
        for record in [selector, model, pipeline.read(self.dest / 'selector' / 'game-0.json')]:
            self.assertEqual(record['model_class'], 'RandomForestClassifier')
            self.assertGreater(record['classifier_fits'], 0)
            self.assertNotIn('logistic_fits', record)
        self.assertNotIn('coef', model)

    def test_stop_marker_blocks_new_fits_and_new_selectors(self):
        (self.dest / 'STOP_REQUESTED').write_text('Unit-test stop marker', encoding='utf-8')
        for operation in [self.fitted, self.selector]:
            with self.assertRaises(SystemExit) as raised:
                operation()
            self.assertEqual(raised.exception.code, 75)

    def test_heldout_perturbation_cannot_change_training_selector(self):
        with fake_models():
            first = self.selector()
        self.data.X.loc[self.validation, 'a'] = 1e8
        self.data.y[self.validation] = 1 - self.data.y[self.validation]
        # Same path proves the input fingerprint also excludes held-out values.
        self.assertEqual(first, self.selector())
        with fake_models():
            repeated = pipeline.build_selector(self.data, self.train, 'toy:train', self.method,
                self.config, self.base, self.dest / 'independent-selector')
        for key in ['selected_ids', 'redundancy', 'paths', 'input_fingerprint']:
            self.assertEqual(first[key], repeated[key])

    def test_utility_original_variable_cache_empty_and_partition_invariants(self):
        method = {**self.method, 'rho': .5}
        game = FeatureUtility(self.data.X, self.data.partition(np.arange(48), 'toy'),
            self.data.numeric, (), method, self.base, 22, 2)
        frozen_folds = game.fold_metadata()
        game.utility(())
        self.assertEqual(game.fit_count, 0)
        self.assertEqual(game.constant_count, 2)
        with fake_models():
            first = game.utility(('b', 'a'))
            self.assertEqual(first, game.utility(('a', 'b')))
        self.assertEqual(game.fit_count, 2)
        self.assertEqual(frozen_folds, game.fold_metadata())
        for fold in game.folds:
            self.assertFalse(set(game.groups[fold['train']]) & set(game.groups[fold['validation']]))
            self.assertTrue(set(fold['selected']) <= set(fold['train']))
        for invalid in [('a', 'a'), ('a=encoded',)]:
            with self.assertRaises(ValueError):
                game.utility(invalid)
        before = game.X.copy()
        original_baseline = deepcopy(game.baseline)
        self.data.X.loc[0, 'a'] = 1e12
        self.base['model']['max_iter'] = 2
        pd.testing.assert_frame_equal(game.X, before)
        self.assertEqual(game.baseline, original_baseline)
        accounting = game.accounting_metadata()
        self.assertTrue(accounting['fit_reconciliation_passed'])
        self.assertTrue(accounting['constant_reconciliation_passed'])
        self.assertTrue(accounting['request_reconciliation_passed'])

    def test_final_model_rejects_heldout_leakage(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            pipeline.fit_evaluate(self.data, self.train, self.train, [0, 37], ('a',),
                                  self.base, 22, self.dest / 'leak.json', config=self.config)
        with self.assertRaisesRegex(ValueError, 'outside training'):
            pipeline.fit_evaluate(self.data, self.train, [0, 37], self.validation, ('a',),
                                  self.base, 22, self.dest / 'leak.json', config=self.config)

    def test_cache_tamper_and_completed_overwrite_are_rejected(self):
        path = self.dest / 'artifact.json'
        pipeline.commit(path, {'value': 1})
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            pipeline.commit(path, {'value': 2})
        path.write_text('{"value": 2}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'modified'):
            pipeline.cached(path)


if __name__ == '__main__':
    unittest.main()
