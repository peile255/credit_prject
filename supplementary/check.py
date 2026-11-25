"""Integration assertions on saved stage outputs; does not train models."""
from pathlib import Path
import argparse,json,unittest
import pandas as pd
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]/('results_supplementary_smoke' if a.smoke else 'results_supplementary')
class Checks(unittest.TestCase):
    def data(self,s):
        path=root/s; self.assertEqual(json.loads((path/'complete.json').read_text())['smoke'],a.smoke);return pd.read_csv(path/'summary.csv')
    def test_s1(self):
        d=self.data('s1');self.assertEqual(len(d),14 if a.smoke else 42);self.assertTrue(d.AP.between(0,1).all());self.assertEqual(d.groupby(['seed','weighted']).model.nunique().min(),7)
    def test_s2(self):
        d=self.data('s2');self.assertEqual(len(d),12 if a.smoke else 120);self.assertTrue(d.Brier.between(0,1).all())
    def test_s3(self):
        d=self.data('s3');self.assertEqual(len(d),96 if a.smoke else 1080);self.assertTrue(d.Overlap.between(0,1).all());self.assertTrue(np.isfinite(d.NormalizedSD).all())
    def test_s4(self):
        d=self.data('s4');self.assertEqual(len(d[d.cohort=='original']),80 if a.smoke else 2000);valid=d[d.coverage==1];self.assertTrue((valid.distance<=valid.radius+1e-10).all());z=valid[valid.method!='RiskReduction'];self.assertTrue((z.new_p<z.threshold).all());self.assertTrue((valid[valid.method=='RiskReduction'].risk_reduction>0).all());self.assertTrue((d[d.coverage==0].holdout_crossing==0).all())
    def test_s5(self):
        d=self.data('s5');self.assertEqual(len(d),32);self.assertTrue((d.holm_p>=d.p-1e-12).all());self.assertTrue(d.holm_p.between(0,1).all())
    def test_s6(self):
        d=self.data('s6');self.assertEqual(len(d),24);self.assertTrue(d.cliffs_delta.between(-1,1).all());self.assertTrue((d.holm_p>=d.p-1e-12).all())
if __name__=='__main__':unittest.main(argv=[__file__])
