import json
from pathlib import Path
import tempfile
import unittest
from flash_v1.localize_v2 import fit
from flash_v1.online_v2 import diagnose,evidence_integrity_reason,COMM_COUNTS,PHASE_CODES


def evidence(scale=None,offset=0):
    scale=scale or {};rows=[]
    for rank in range(4):
        for step in range(6):
            for (slot,phase),count in COMM_COUNTS.items():
                duration=100*scale.get(rank,1)
                intervals=[dict(start_us=i*10,end_us=(i+1)*10,phase_code=256|PHASE_CODES[slot,phase]) for i in range(count)]
                rows.append(dict(record_type='measurement',rank=rank,iteration=step+offset,epoch=1,
                    slot=slot,phase=phase,layer=0,dtype='torch.bfloat16',shape=[16,896],
                    stream_id=0,measurement='local_residual',coverage_complete=True,
                    duration_us=duration,module_elapsed_us=duration+count*10,
                    communication_intervals=intervals))
    return rows


class RedisStub:
    def __init__(self,records):self.records=records
    def lrange(self,key,*_):return [json.dumps(row) for row in self.records] if key.endswith('0') else []


class OnlineV2Tests(unittest.TestCase):
    def setUp(self):
        self.model=fit(evidence(),[evidence(offset=100)],fit_sessions=['fit'],calibration_sessions=['cal'])
        self.schedule=dict(epoch=1,start=0,end=5)
    def test_completion_and_unique_local_anomaly(self):
        rows=evidence({2:4})
        markers=[dict(record_type='iteration_complete',epoch=1,rank=rank,iteration=iteration,rows_count=4) for rank in range(4) for iteration in range(6)]
        with tempfile.TemporaryDirectory() as output:
            Path(output,'topology.json').write_text(json.dumps({'world_size':4}))
            self.assertIsNone(diagnose(RedisStub(rows+markers[:-1]),output,self.schedule,self.model))
            self.assertEqual(diagnose(RedisStub(rows+markers),output,self.schedule,self.model)['predicted'],'rank:2')
            self.assertEqual(diagnose(RedisStub(rows[:-1]+markers),output,self.schedule,self.model)['reason'],'completion_row_count_mismatch')
    def test_linear_model_dispatch_preserves_transport_and_integrity(self):
        from flash_v1 import localize_v2_linear
        model=localize_v2_linear.fit(evidence(),[evidence(offset=100)],fit_sessions=['fit'],calibration_sessions=['cal'])
        rows=evidence({2:4})
        markers=[dict(record_type='iteration_complete',epoch=1,rank=rank,iteration=step,rows_count=4) for rank in range(4) for step in range(6)]
        with tempfile.TemporaryDirectory() as output:
            Path(output,'topology.json').write_text(json.dumps({'world_size':4}))
            self.assertEqual(diagnose(RedisStub(rows+markers),output,self.schedule,model)['predicted'],'rank:2')
    def test_missing_role_is_invalid_even_when_marker_count_matches(self):
        rows=evidence();rows.pop()
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'incomplete_or_duplicate_semantic_roles')
    def test_invalid_stream_scope_and_nonfinite_duration(self):
        rows=evidence();rows[0]['coverage_complete']=False
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'incomplete_or_invalid_scope')
        rows=evidence();rows[0]['module_elapsed_us']=float('nan')
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'nonfinite_or_invalid_scope_duration')
    def test_bad_union_count_and_semantic_phase(self):
        rows=evidence();rows[0]['duration_us']+=1
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'local_residual_union_mismatch')
        rows=evidence();rows[0]['communication_intervals']=[]
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'incomplete_communication_coverage')
        rows=evidence();rows[0]['communication_intervals'][0]['phase_code']=999
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'communication_semantic_phase_mismatch')
        rows=evidence();rows[0]['communication_intervals'][0]['end_us']=1000
        self.assertEqual(evidence_integrity_reason(rows,self.schedule,range(4)),'communication_interval_outside_scope')

if __name__=='__main__':unittest.main()
