import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from flash_v1 import localize_v2_linear as linear
from flash_v1.online_v2 import diagnose
from flash_v1 import stage_forward_localize as _auxiliary_test_module
from flash_v1.tests.test_profile_transition import evidence


class Client:
    def __init__(self, original, auxiliary):self.original=original;self.auxiliary=auxiliary;self.aux_reads=0
    def lrange(self,key,*args):
        if 'stage_forward_aux' in key:
            self.aux_reads+=1;rows=self.auxiliary if key.endswith('0') else []
        else:rows=self.original if key.endswith('0') else []
        return [json.dumps(r) for r in rows]


class FallbackTests(unittest.TestCase):
    def test_accepted_dc89_keeps_hint_and_never_reads_auxiliary(self):
        self.run_case(True)
    def test_auxiliary_localized_does_not_feed_continuation_candidate(self):
        self.run_case(False)
    def test_peer_auxiliary_does_not_feed_continuation_candidate(self):
        self.run_case(False, peer=True)
    def run_case(self, accepted, peer=False):
        healthy=evidence(onset=100);rows=evidence(onset=0) if accepted else healthy
        model=linear.fit(healthy,[healthy],fit_sessions=['fit'],calibration_sessions=['cal'])
        markers=[dict(record_type='iteration_complete',epoch=1,rank=r,iteration=i,rows_count=4) for i in range(8) for r in range(4)]
        aux=[dict(record_type='stage_forward_aux',epoch=1,rank=r,iteration=i) for i in range(8) for r in range(4)]
        client=Client(rows+markers,aux);seen=[]
        def hint(observations, frozen, ranks, schedule, result):
            seen.append(result['status']);return {'rank':0} if accepted else None
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'topology.json').write_text(json.dumps(dict(world_size=4)))
            path=root/'aux_model.json';path.write_text(json.dumps(dict(sha256='frozen-test',model_type='flash_stage_forward_tp_peer_v1' if peer else 'flash_stage_forward_healthy_v1')))
            (root/'configuration.json').write_text(json.dumps(dict(monitor_profile_transition=True,stage_forward_aux_enabled=True,stage_forward_aux_model=str(path),stage_forward_aux_context_binding=str(path))))
            fake_peer=SimpleNamespace(infer_bound=lambda *args:dict(status='localized',rank=2,profile_transition={'rank':2}),verify_topology=lambda *args:None)
            with patch.dict('sys.modules',{'flash_v1.stage_forward_peer':fake_peer}),patch('flash_v1.profile_transition.candidate',side_effect=hint),patch('flash_v1.stage_forward_localize.infer_bound',create=True,return_value=dict(status='localized',rank=2,profile_transition={'rank':2})):
                result=diagnose(client,root,dict(epoch=1,start=0,end=7),model)
            original=json.loads((root/'online_profile_1/original_dc89.json').read_text())
        self.assertEqual(seen,['localized' if accepted else 'rejected'])
        if accepted:
            self.assertEqual(result['rank'],0);self.assertEqual(result['profile_transition'],{'rank':0});self.assertEqual(client.aux_reads,0)
        else:
            self.assertEqual(result['rank'],2);self.assertEqual(result['source'],'stage_forward_aux')
            self.assertNotIn('profile_transition',result)
            self.assertEqual(original['status'],'rejected');self.assertIsNone(original['rank'])

if __name__=='__main__':unittest.main()
