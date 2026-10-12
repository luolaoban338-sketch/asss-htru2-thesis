import unittest
import numpy as np
import pandas as pd
from experiments.feature_path import redundancy_matrix, forward_path


class FeaturePathTests(unittest.TestCase):
    def test_duplicate_numeric_constant_and_categorical_blocks(self):
        X = pd.DataFrame({'x': [0., 1., 2., 3.], 'copy': [0., 2., 4., 6.],
            'constant': [1.] * 4, 'cat': ['a', 'a', 'b', 'b']})
        R = redundancy_matrix(X, ['x', 'copy', 'constant'], ['cat'])
        np.testing.assert_allclose(R, R.T)
        self.assertEqual(R.shape, (4, 4)); self.assertAlmostEqual(R[0, 1], 1)
        np.testing.assert_array_equal(R[2], [0, 0, 0, 0])
        self.assertAlmostEqual(R[0, 3], 2 / np.sqrt(5))

    def test_eta_zero_and_negative_scores_keep_full_path(self):
        path = forward_path(['a', 'b', 'c'], [-1, -1, -2], np.zeros((3, 3)), 0)
        self.assertEqual([r['added'] for r in path], ['a', 'b', 'c'])
        self.assertEqual(path[-1]['features'], ['a', 'b', 'c'])
        self.assertEqual(path[-1]['objective'], -4)

    def test_redundancy_changes_order_and_matches_objective(self):
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.]])
        scores = np.array([.5, .4, .3]); features = ['a', 'b', 'c']
        path = forward_path(features, scores, R, .6)
        self.assertEqual([r['added'] for r in path], ['a', 'c', 'b'])
        for r in path:
            ids = [features.index(f) for f in r['features']]
            expected = scores[ids].sum() - .6 / 3 * sum(R[i, j] for i in ids for j in ids if i < j)
            self.assertAlmostEqual(r['objective'], expected)

    def test_invalid_association_rejected(self):
        for R in [np.ones((2, 2)), np.array([[0, 1], [0, 0]]), np.array([[0, np.nan], [np.nan, 0]])]:
            with self.assertRaises(ValueError): forward_path(['a', 'b'], [1, 2], R, .5)


if __name__ == '__main__': unittest.main()
