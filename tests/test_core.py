import unittest
import numpy as np
from src.explain import permutation_shap,aggregate
from src.evaluation import stability,calibration_bins
from run_all import split,load
class Linear:
    def predict_proba(self,X):
        p=.2+.1*X[:,0]+.2*X[:,1];return np.c_[1-p,p]
class TestCore(unittest.TestCase):
    def test_data_and_split(self):
        df,X,y,g=load(False);self.assertEqual(X.shape,(30000,23));self.assertEqual(int(y.sum()),6636);self.assertEqual(int(np.isnan(X).sum()),0)
        tr,ca,te=split(X,y,g,42);self.assertEqual(len(tr)+len(ca)+len(te),len(y));self.assertFalse(set(g[tr])&set(g[ca]));self.assertFalse(set(g[te])&set(g[ca]));self.assertFalse(set(g[tr])&set(g[te]))
    def test_shap_linear(self):
        x=np.array([[1.,1.],[0.,1.]]);bg=np.array([[0.,0.]]);phi=permutation_shap(Linear(),x,bg,2,42);np.testing.assert_allclose(phi,[[.1,.2],[0,.2]],atol=1e-9)
    def test_stability(self):
        a=np.array([[[.1,.3,-.2]],[[.1,.3,-.2]],[[.1,.3,-.2]]]);s=stability(a,2)[0];self.assertAlmostEqual(s['Overlap'],1);self.assertAlmostEqual(s['Rank'],1);self.assertAlmostEqual(s['Direction'],1);self.assertEqual(aggregate(a,2)['level'][0,1],'High')
    def test_bins(self):
        b=calibration_bins(np.array([0,1]),np.array([0.,1.]));self.assertEqual(sum(x['n'] for x in b),2)
if __name__=='__main__':unittest.main()
