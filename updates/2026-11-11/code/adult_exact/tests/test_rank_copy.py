"""Exercise real redundancy arithmetic with synthetic encoded data; no fits."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from private_science import engine

class RankCopyRegression(unittest.TestCase):
    def test_actual_redundancy_function(self):
        fixtures=[[[1,2,3]]*4, [[1,2,5],[1,3,5],[2,3,5],[4,9,5]],
                  [[-4,8,0],[-1,3,4],[0,-3,2],[7,2,-4],[8,0,1]],
                  [[3,2,1]], [[1],[1],[3],[9]]]
        for values in fixtures:
            with self.subTest(values=values):
                z=np.asarray(values,dtype=float)
                cols=[f'x{i}' for i in range(z.shape[1])]
                data=SimpleNamespace(X=pd.DataFrame(z,columns=cols))
                enc=SimpleNamespace(blocks_={c:[i] for i,c in enumerate(cols)},diagnostics=lambda x:{})
                with patch.object(engine,'encoder_for',return_value=enc), \
                     patch.object(engine,'checked_fit',return_value=(enc,0.0)), \
                     patch.object(engine,'checked_transform',return_value=z):
                    actual,_=engine.redundancy(data,np.arange(len(z)),object(),{'phase':'synthetic_regression'})
                ranks=np.asarray([[1+sum(v<value for v in z[:,j])+(sum(v==value for v in z[:,j])-1)/2
                                  for j,value in enumerate(row)] for row in z])
                centered=ranks-ranks.mean(axis=0)
                expected=np.zeros_like(actual)
                for i in range(len(cols)):
                    for j in range(i):
                        denominator=np.linalg.norm(centered[:,i])*np.linalg.norm(centered[:,j])
                        value=abs(sum(centered[:,i]*centered[:,j]))/denominator if denominator else 0
                        expected[i,j]=expected[j,i]=min(1,value)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(z,np.asarray(values,dtype=float))

if __name__=='__main__':unittest.main()
