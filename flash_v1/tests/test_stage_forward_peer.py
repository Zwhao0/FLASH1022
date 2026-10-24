"""CPU-only protocol tests for current-work TP2 contrast and physical scales."""
import copy
import unittest

from flash_v1.stage_forward_peer import (context_binding, fallback, fit, infer_bound,
    verify_topology, _hash)
from flash_v1.tests.test_stage_forward_localize import sample


class PeerTests(unittest.TestCase):
    def setUp(self):
        self.pairs=[[0,2],[1,3],[4,6],[5,7],[8,10],[9,11],[12,14],[13,15]]
        self.topology=dict(world_size=16,parallel_layout='legacy',
            groups=[dict(name='tensor_parallel',members=pair) for pair in self.pairs])
        rows={phase:[sample(rank,step) for rank in range(16) for step in range(start,end+1)]
              for phase,start,end in [('fit',85,92),('calibration',104,111)]}
        for values in rows.values():
            for row in values:
                row['stage_metadata']['instrumentation_context']='plain'
                row['stage_metadata']['tp_group_ranks']=next(pair for pair in self.pairs if row['rank'] in pair)
        sources=dict(fit=dict(session_id='cost',seed=7,healthy=True,layout='legacy',start=85,end=92),
            calibration=dict(session_id='cost',seed=7,healthy=True,layout='legacy',start=104,end=111))
        manifest=dict(protocol='plain_cost_fit8_cal8_v1',expected_ranks=list(range(16)),window_size=8,
                      by_layout={'legacy':sources})
        self.model,self.audit=fit(rows,manifest,{'legacy':self.topology},'plain')
        self.binding=context_binding(self.model,{'test':'known static plain observer and actual TP partition'})
        self.rows=copy.deepcopy(rows['calibration'])

    def infer(self,rows=None):
        return infer_bound(self.rows if rows is None else rows,self.model,list(range(16)),self.binding)

    def bump(self,rank,extra,steps=None):
        for row in self.rows:
            if row['rank']==rank and (steps is None or row['iteration'] in steps):
                row['duration_us']+=extra
                for mb in row['microbatches']:
                    mb['duration_us']+=extra/2
                    mb['forward_elapsed_us']+=extra/2

    def test_physical_floor_uses_pair_mean_not_near_zero_difference(self):
        baseline=self.model['layouts']['legacy']['baselines']['0,2']
        self.assertEqual(baseline['offset_us'],0.)
        self.assertEqual(baseline['mad_difference_us'],0.)
        self.assertEqual(baseline['fit_pair_mean_us'],180.)
        self.assertEqual(baseline['scale_us'],27.)
        self.assertEqual(self.infer()['status'],'rejected')
        verify_topology(self.model,self.topology,'legacy')
        other=copy.deepcopy(self.topology)
        other['groups'][0]['members']=[0,1]
        with self.assertRaises(ValueError):verify_topology(self.model,other,'legacy')

    def test_three_matched_persistent_steps_and_unknown_current_work(self):
        for row in self.rows:
            if row['rank']==0 and row['iteration']<109:
                row['microbatches'][0]['work']['0:experts.0'][0]['shape'][0]=7
        self.bump(0,300,steps={109,110,111})
        result=self.infer()
        self.assertEqual((result['status'],result['rank']),('localized',0))
        self.assertEqual(result['candidate_pair'],'0,2')
        self.assertEqual(result['pair_support']['0,2']['matched_iterations'],3)
        self.assertEqual(result['pair_support']['0,2']['above_threshold_same_direction_iterations'],3)
        self.assertIn('0,2',result['unknown'])

    def test_three_directional_steps_must_each_cross_threshold(self):
        for row in self.rows:
            if row['rank']==0 and row['iteration']<109:
                row['microbatches'][0]['work']['0:experts.0'][0]['shape'][0]=7
        self.bump(0,300,steps={109,110})
        result=self.infer()
        self.assertGreater(result['pair_scores']['0,2'],6)
        self.assertEqual(result['pair_support']['0,2']['above_threshold_same_direction_iterations'],2)
        self.assertEqual(result['reason'],'insufficient_persistent_directional_threshold_support')
        self.assertIsNone(result['rank'])

    def test_current_matched_work_needs_no_historical_token_range(self):
        for row in self.rows:
            if row['rank'] in (0,2):
                row['microbatches'][0]['work']['0:experts.0'][0]['shape'][0]=7
        self.bump(2,300)
        result=self.infer()
        self.assertEqual(result['rank'],2)
        self.assertEqual(result['pair_support']['0,2']['matched_iterations'],8)

    def test_pair_runner_up_is_real_and_unknown_pairs_are_not_zero(self):
        self.bump(0,300)
        self.bump(1,270)
        result=self.infer()
        self.assertEqual(result['runner_up_pair'],'1,3')
        self.assertLess(result['actual_pair_margin'],2)
        self.assertEqual(result['status'],'rejected')
        for row in self.rows:
            if row['rank'] in [pair[0] for pair in self.pairs[1:]]:
                row['microbatches'][1]['work']['0:shared_expert'][0]['shape'][0]=17
        result=self.infer()
        self.assertEqual(result['reason'],'fewer_than_two_comparable_pairs')
        self.assertEqual(set(result['pair_scores']),{'0,2'})

    def test_context_integrity_completeness_and_model_hash_fail_closed(self):
        self.rows[0]['stage_metadata']['instrumentation_context']='scoped_dispatch'
        self.assertEqual(self.infer()['reason'],'missing_or_incompatible_instrumentation_context')
        self.rows[0]['stage_metadata']['instrumentation_context']='plain'
        original_pair=self.rows[0]['stage_metadata'].pop('tp_group_ranks')
        self.assertEqual(self.infer()['reason'],'missing_or_invalid_observed_tp_group_ranks')
        self.rows[0]['stage_metadata']['tp_group_ranks']=[0,1]
        self.assertEqual(self.infer()['reason'],'declared_current_tp_group_mismatch')
        self.rows[0]['stage_metadata']['tp_group_ranks']=original_pair
        self.assertEqual(self.infer(self.rows[:-1])['status'],'rejected')
        self.rows[0]['microbatches'][0]['communication_intervals'][0]['end_us']=101
        self.assertEqual(self.infer()['status'],'rejected')
        self.model['relative_floor']=.14
        self.model['sha256']=_hash({key:value for key,value in self.model.items() if key!='sha256'})
        with self.assertRaises(ValueError):self.infer()

    def test_original_dc89_localized_is_never_read_or_changed(self):
        original=dict(status='localized',rank=12,reason='original')
        self.assertIs(fallback(original,None,None,None,None),original)


if __name__=='__main__':unittest.main()
