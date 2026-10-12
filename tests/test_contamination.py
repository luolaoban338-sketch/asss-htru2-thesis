import unittest
import numpy as np
import pandas as pd
from experiments.contamination import contaminate


class ContaminationTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({'x': np.arange(100.), 'z': np.arange(100.) ** 2,
                               'constant': np.ones(100), 'cat': ['a', 'b'] * 50})
        self.ids = np.array([f'r{i:03}' for i in range(100)])
        self.numeric = ['x', 'z', 'constant']

    def test_exact_cells_and_magnitude_and_input_unchanged(self):
        before = self.X.copy(deep=True)
        out, meta = contaminate(self.X, self.ids, self.numeric, .05, 7)
        pd.testing.assert_frame_equal(self.X, before)
        self.assertEqual(int((out != before).to_numpy().sum()), 5)
        self.assertEqual(len(set(e['row_id'] for e in meta['events'])), 5)
        self.assertNotIn('constant', meta['eligible_iqr'])
        for e in meta['events']:
            self.assertAlmostEqual(e['after'] - e['before'], e['sign'] * 10 * e['iqr'])
        pd.testing.assert_series_equal(out.cat, self.X.cat)

    def test_zero_is_identity(self):
        out, meta = contaminate(self.X, self.ids, self.numeric, 0, 7)
        pd.testing.assert_frame_equal(out, self.X)
        self.assertEqual(meta['count'], 0)

    def test_nested_rates_share_cells_and_signs(self):
        _, a = contaminate(self.X, self.ids, self.numeric, .05, 7)
        _, b = contaminate(self.X, self.ids, self.numeric, .10, 7)
        self.assertEqual(a['events'], b['events'][:5])

    def test_reproducible_and_different_draws(self):
        a, ma = contaminate(self.X, self.ids, self.numeric, .10, 7)
        b, mb = contaminate(self.X, self.ids, self.numeric, .10, 7)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(ma, mb)
        _, mc = contaminate(self.X, self.ids, self.numeric, .10, 8)
        self.assertNotEqual(ma['events'], mc['events'])

    def test_reordering_preserves_identity_draw(self):
        a, _ = contaminate(self.X, self.ids, self.numeric, .10, 7)
        b, _ = contaminate(self.X.iloc[::-1], self.ids[::-1], self.numeric, .10, 7)
        pd.testing.assert_frame_equal(a, b.iloc[::-1])

    def test_missing_not_imputed_or_corrupted(self):
        X = self.X.copy(); X.loc[:49, 'x'] = np.nan
        out, m = contaminate(X, self.ids, self.numeric, 1, 7)
        np.testing.assert_array_equal(X.isna(), out.isna())
        self.assertTrue(all(pd.notna(X.iloc[e['local_position']][e['feature']]) for e in m['events']))

    def test_invalid_and_infeasible_inputs(self):
        for rate in [-.1, 1.1, np.nan]:
            with self.assertRaises(ValueError): contaminate(self.X, self.ids, self.numeric, rate, 7)
        with self.assertRaises(ValueError): contaminate(self.X, self.ids, ['constant'], .1, 7)
        with self.assertRaises(ValueError): contaminate(self.X, ['same'] * 100, self.numeric, .1, 7)

    def test_round_up(self):
        _, meta = contaminate(self.X.iloc[:21], self.ids[:21], self.numeric, .05, 7)
        self.assertEqual(meta['count'], 2)


if __name__ == '__main__': unittest.main()
