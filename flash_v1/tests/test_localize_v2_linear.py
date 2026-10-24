import unittest
from flash_v1 import localize_v2_linear as linear
from flash_v1.tests.test_localize_v2 import rows

class LinearTests(unittest.TestCase):
    def setUp(self):self.model=linear.fit(rows(),[rows(offset=100)],fit_sessions=['fit'],calibration_sessions=['cal'])
    def test_additive_scale_and_unique_vs_ambiguous(self):
        self.assertEqual(next(iter(self.model['baselines'].values()))['scale_us'],15.)
        self.assertEqual(linear.infer(rows({2:4}),self.model,range(4))['rank'],2)
        self.assertIsNone(linear.infer(rows({1:4,2:4}),self.model,range(4))['rank'])
    def test_rank_permutation_and_model_integrity(self):
        sample=rows({2:4});mapping={0:10,1:9,2:7,3:1}
        for row in sample:row['rank']=mapping[row['rank']]
        self.assertEqual(linear.infer(sample,self.model,mapping.values())['rank'],7)
        linear.verify_model(self.model)
        changed=dict(self.model,min_score=3.)
        with self.assertRaises(ValueError):linear.verify_model(changed)
    def test_unknown_shape_truth_and_same_session_rejected(self):
        sample=rows({2:4})
        for row in sample:
            if row['rank']==2:row['shape']=[999]
        self.assertEqual(linear.infer(sample,self.model,range(4))['reason'],'incomplete_comparable_evidence')
        sample=rows();sample[0]['target_ranks']=[1]
        with self.assertRaises(ValueError):linear.infer(sample,self.model,range(4))
        with self.assertRaises(ValueError):linear.fit(rows(),[rows()],fit_sessions=['same'],calibration_sessions=['same'])

if __name__=='__main__':unittest.main()
