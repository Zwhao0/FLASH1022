import unittest
from flash_v1.localize_v2 import fit,infer,local_residual


def rows(scale=None,offset=0):
    scale=scale or {}
    return [dict(rank=r,iteration=i+offset,measurement='local_residual',layer=1,phase='bwd',slot='attention',shape=[16,896],duration_us=100*scale.get(r,1),coverage_complete=True) for r in range(4) for i in range(6)]


class V2Tests(unittest.TestCase):
    def setUp(self):self.model=fit(rows(),[rows(offset=100)],fit_sessions=['fit'],calibration_sessions=['cal'])
    def test_disjoint_health_sessions_required(self):
        with self.assertRaises(ValueError):fit(rows(),[rows()],fit_sessions=['same'],calibration_sessions=['same'])
    def test_unique_local_slowdown_and_health_rejection(self):
        self.assertEqual(infer(rows({2:4}),self.model,range(4))['rank'],2)
        self.assertEqual(infer(rows(),self.model,range(4))['status'],'rejected')
    def test_wait_union_excludes_propagated_communication(self):
        self.assertEqual(local_residual(0,1000,[(100,700),(600,900)]),200)
        with self.assertRaises(ValueError):local_residual(0,100,[(0,110)])
    def test_rank_permutation_and_ambiguous_rejection(self):
        sample=rows({2:4});mapping={0:9,1:7,2:4,3:2}
        for row in sample:row['rank']=mapping[row['rank']]
        self.assertEqual(infer(sample,self.model,mapping.values())['rank'],4)
        self.assertIsNone(infer(rows({1:4,2:4}),self.model,range(4))['rank'])
    def test_healthy_token_interpolation_is_bounded_and_rank_free(self):
        train=[]
        for tokens in (0,32):
            sample=rows()
            for row in sample:row.update(slot='experts',shape=[tokens,896],duration_us=100+tokens*10)
            train.extend(sample)
        cal=rows()
        for row in cal:row.update(slot='experts',shape=[16,896],duration_us=260)
        model=fit(train,[cal],fit_sessions=['fit'],calibration_sessions=['cal'])
        self.assertEqual(infer(cal,model,range(4))['status'],'rejected')
        self.assertEqual(infer(cal,model,range(4))['diagnostics']['interpolated_semantic_shape'],24)
        for row in cal:
            if row['rank']==2:row['duration_us']*=4
        self.assertEqual(infer(cal,model,range(4))['rank'],2)
        for row in cal:
            if row['rank']==2:row['shape']=[33,896]
        self.assertEqual(infer(cal,model,range(4))['reason'],'incomplete_comparable_evidence')
    def test_unknown_shape_incomplete_or_truth_input(self):
        sample=rows({2:4})
        for row in sample:
            if row['rank']==2:row['shape']=[123,896]
        self.assertEqual(infer(sample,self.model,range(4))['reason'],'incomplete_comparable_evidence')
        sample=rows();sample[0]['fault_mode']='anything'
        with self.assertRaises(ValueError):infer(sample,self.model,range(4))

if __name__=='__main__':unittest.main()
