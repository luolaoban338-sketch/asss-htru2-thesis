import unittest
import numpy as np
import pandas as pd
from src.data import ROOT, load_dataset
from src.evaluation import classification_metrics

class HTRU2MigrationTests(unittest.TestCase):
    def test_official_values_and_explicit_schema(self):
        data=load_dataset('htru2')
        raw=pd.read_csv(ROOT/'data/raw/HTRU_2.csv',header=None)
        self.assertEqual(data.X.shape,(17898,8))
        self.assertEqual(int(data.y.sum()),1639)
        self.assertEqual(list(data.X.columns),['Profile_mean','Profile_stdev','Profile_skewness','Profile_kurtosis','DM_mean','DM_stdev','DM_skewness','DM_kurtosis'])
        np.testing.assert_allclose(data.X.to_numpy(),raw.iloc[:,:8].to_numpy(),rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(data.y,raw.iloc[:,8].to_numpy())
        self.assertEqual(len(set(data.group_ids)),17898)
        self.assertEqual(data.categorical,())
        with self.assertRaises(ValueError): load_dataset('compas')

    def test_average_precision_independent_hand_case(self):
        metrics=classification_metrics([0,1,0,1],[.9,.8,.7,.6])
        self.assertEqual(metrics['average_precision'],.5)
        self.assertIsNone(classification_metrics([0,0],[.2,.3])['average_precision'])
