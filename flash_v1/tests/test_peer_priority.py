"""Peer fusion, bounded missing-sidecar fallback and independent core continuation."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from flash_v1 import online,online_v2,localize_v2_linear as linear
from flash_v1.tests.test_profile_transition import evidence

PEER='flash_stage_forward_tp_peer_v1'
class Client:
    def __init__(self,records,aux):self.records=records;self.aux=aux;self.reads=0
    def lrange(self,key,*args):
        if 'stage_forward_aux' in key:self.reads+=1;rows=self.aux
        else:rows=self.records
        return [json.dumps(x) for x in rows] if key.endswith('0') else []

class PeerPriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        healthy=evidence(onset=100)
        cls.model=linear.fit(healthy,[healthy],fit_sessions=['fit'],calibration_sessions=['cal'])
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.schedule=dict(epoch=1,start=2,end=9)
        (self.root/'topology.json').write_text(json.dumps(dict(world_size=4)))
        self.path=self.root/'aux.json';self.path.write_text(json.dumps(dict(sha256='test',model_type=PEER)))
        self.cfg=dict(monitor_profile_transition=True,training_steps=100,stage_forward_aux_enabled=True,
                      stage_forward_aux_model=str(self.path),stage_forward_aux_context_binding=str(self.path))
        self.configure()
        self.aux=[dict(record_type='stage_forward_aux',epoch=1,rank=r,iteration=i) for i in range(2,10) for r in range(4)]
    def configure(self): (self.root/'configuration.json').write_text(json.dumps(self.cfg))
    def client(self,accepted=True,aux=None):
        rows=[dict(r,iteration=r['iteration']+2) for r in evidence(onset=0 if accepted else 100)]
        markers=[dict(record_type='iteration_complete',epoch=1,rank=r,iteration=i,rows_count=4) for i in range(2,10) for r in range(4)]
        return Client(rows+markers,self.aux if aux is None else aux)
    def call(self,client,peer=None,topology_error=None):
        self.seen=[]
        def candidate(rows,model,expected,schedule,result):
            self.seen.append(copy.deepcopy(result))
            return {'rank':0} if result['status']=='localized' else None
        fake=SimpleNamespace(infer_bound=lambda *a: peer or dict(status='localized',rank=2,
                            profile_transition={'rank':2},profile_transition_core={'rank':2}),
                            verify_topology=lambda *a:None)
        if topology_error:fake.verify_topology=lambda *a:(_ for _ in ()).throw(ValueError(topology_error))
        with patch.dict('sys.modules',{'flash_v1.stage_forward_peer':fake}),patch('flash_v1.profile_transition.candidate',side_effect=candidate):
            return online_v2.diagnose(client,self.root,self.schedule,self.model)
    def test_peer_priority_preserves_original_hint_and_ack_identity(self):
        result=self.call(self.client());self.assertEqual(result['rank'],2)
        self.assertEqual(result['reason'],'stage_forward_tp_peer_priority')
        self.assertEqual(self.seen[0]['rank'],0)
        self.assertEqual(result['profile_transition_core'],dict(status='localized',rank=0,profile_transition={'rank':0}))
        old=json.loads((self.root/'online_profile_1/original_dc89.json').read_text());self.assertEqual(old['rank'],0)
        pending=dict(schedule=self.schedule)
        event=online.profile_transition_event(self.cfg,pending,result,1,4,10,None)
        self.assertEqual(event['transition'],{'rank':0});self.assertEqual(event['parent_epoch'],1)
        self.assertIsNone(online.profile_transition_event(self.cfg,dict(pending,parent_epoch=1),result,2,4,20,None))
    def test_peer_rejected_retains_accepted_core(self):
        result=self.call(self.client(),dict(status='rejected',rank=None,reason='support'))
        self.assertEqual((result['status'],result['rank']),('localized',0));self.assertEqual(result['profile_transition'],{'rank':0})
    def test_auxiliary_cannot_invent_core_hint(self):
        result=self.call(self.client(False));self.assertEqual(result['rank'],2)
        self.assertNotIn('profile_transition',result);self.assertNotIn('profile_transition_core',result)
        self.assertIsNone(online.profile_transition_event(self.cfg,dict(schedule=self.schedule),result,1,4,10,None))
    def test_late_rows_can_complete_before_deadline(self):
        client=self.client(aux=self.aux[:-1]);self.assertIsNone(self.call(client))
        client.aux=self.aux;self.assertEqual(self.call(client)['rank'],2)
    def test_missing_aux_deadline_returns_accepted_core(self):
        self.assertIsNone(self.call(self.client(aux=[])))
        with patch.object(online,'diagnose',online_v2.diagnose):
            result=online.bounded_diagnostic_timeout(None,self.root,self.schedule,self.model)
        self.assertEqual((result['status'],result['rank'],result['predicted']),('localized',0,'rank:0'))
        self.assertEqual(result['stage_forward_aux']['reason'],'bounded_auxiliary_drain_timeout')
        self.assertEqual(result['profile_transition'],{'rank':0})
    def test_missing_aux_deadline_preserves_rejected_core(self):
        self.assertIsNone(self.call(self.client(False,[])))
        result=online_v2.diagnose_timeout(None,self.root,self.schedule,self.model)
        self.assertEqual((result['status'],result['rank']),('rejected',None));self.assertEqual(result['predicted'],'inconclusive')
    def test_invalid_or_duplicate_aux_falls_back_immediately(self):
        for aux in [[{'malformed':True}],self.aux+[self.aux[0]]]:
            with self.subTest(aux_size=len(aux)):
                result=self.call(self.client(aux=aux));self.assertEqual(result['rank'],0)
                self.assertEqual(result['stage_forward_aux']['status'],'rejected')
    def test_invalid_topology_or_context_retains_core(self):
        self.assertEqual(self.call(self.client(),topology_error='wrong_actual_tp_group')['rank'],0)
        result=self.call(self.client(),dict(status='rejected',rank=None,reason='instrumentation_context_mismatch'))
        self.assertEqual(result['rank'],0)
    def test_disabled_aux_keeps_old_path(self):
        self.cfg['stage_forward_aux_enabled']=False;self.configure();client=self.client()
        result=self.call(client);self.assertEqual(result['rank'],0);self.assertEqual(client.reads,0)
    def test_timeout_missing_or_stale_core_does_not_create_verdict(self):
        self.assertIsNone(online_v2.diagnose_timeout(None,self.root,self.schedule,self.model))
        self.call(self.client(aux=[]));p=self.root/'online_profile_1/original_dc89.json';original=json.loads(p.read_text())
        for change in [dict(frozen_model_hash='other'),dict(run_id='online_profile_2'),dict(rank=True)]:
            p.write_text(json.dumps(dict(original,**change)))
            self.assertIsNone(online_v2.diagnose_timeout(None,self.root,self.schedule,self.model))
    def test_v1_timeout_still_uses_original_generic_result(self):
        result=online.bounded_diagnostic_timeout(None,self.root,self.schedule,self.model)
        self.assertEqual(result,dict(predicted='inconclusive',reason='bounded_drain_or_coverage_timeout'))
    def test_absolute_auxiliary_keeps_historical_core_priority(self):
        core=dict(status='localized',rank=0)
        self.assertIs(online_v2._combine_stage_forward(core,dict(status='localized',rank=2),'flash_stage_forward_healthy_v1'),core)
    def test_missing_binding_or_wrong_window_preserves_core_without_wait(self):
        self.cfg.pop('stage_forward_aux_context_binding');self.configure()
        self.assertEqual(self.call(self.client())['rank'],0)
        self.cfg['stage_forward_aux_context_binding']=str(self.path);self.configure()
        self.schedule['end']=8
        self.assertEqual(self.call(self.client())['rank'],0)
    def test_missing_model_file_preserves_core_immediately(self):
        self.path.unlink();result=self.call(self.client())
        self.assertEqual(result['rank'],0);self.assertEqual(result['stage_forward_aux']['status'],'rejected')
    def test_invalid_model_record_preserves_accepted_core(self):
        self.path.write_text('[]');result=self.call(self.client())
        self.assertEqual(result['rank'],0)
        self.assertEqual(result['stage_forward_aux']['reason'],'invalid_auxiliary_model_record')
    def test_actual_controller_deadline_preserves_core_once(self):
        fixture=self.client(aux=[]);clock=[0.];calls=[]
        class Training:
            def __init__(self):self.values={};self.polls=0;self.operations=[]
            def exists(self,key):return key=='shutdown' and 'online_completed_1' in self.values
            def get(self,key):return self.values.get(key)
            def set(self,key,value):
                self.operations.append((key,clock[0]));self.values[key]=value
                if key=='online_request':self.values['online_schedule_1']=json.dumps(dict(epoch=1,start=2,end=9))
            def lrange(self,key,start,end):
                if key.startswith('online_evidence') or key.startswith('online_stage_forward_aux'):return fixture.lrange(key,start,end)
                if start==-1:return [json.dumps(dict(iteration=10))]
                self.polls+=1;last=1 if self.polls==1 else 10
                return [json.dumps(dict(iteration=i,wall_iteration_ms=100,profile_epoch=1 if 2<=i<=9 else 0)) for i in range(1,last+1)]
        class Detector:
            def __init__(self,warmup):pass
            def add(self,iteration,duration):return dict(iteration=iteration)
        def diagnose(*args):calls.append(clock[0]);return online_v2.diagnose(*args)
        diagnose.timeout_result=online_v2.diagnose_timeout
        fake=SimpleNamespace(infer_bound=lambda *a:dict(status='localized',rank=2),verify_topology=lambda *a:None)
        client=Training()
        with patch.dict('sys.modules',{'flash_v1.stage_forward_peer':fake}),patch('flash_v1.profile_transition.candidate',return_value={'rank':0}),patch.object(online,'diagnose',diagnose),patch.object(online,'WindowDetector',Detector),patch.object(online,'drains_after',return_value=True),patch.object(online.time,'monotonic',side_effect=lambda:clock[0]),patch.object(online.time,'sleep',side_effect=lambda _:clock.__setitem__(0,clock[0]+1)):
            online.controller(client,self.root,self.model,max_profiles=1)
        result=json.loads(client.values['online_completed_1']);self.assertEqual(result['rank'],0)
        ack=[t for k,t in client.operations if k=='online_completed_1'];self.assertEqual(ack,[calls[0]+10])
        self.assertEqual(sum(k=='online_request' for k,t in client.operations),1)
        self.assertEqual(client.values['online_pending'],'0')
        self.assertLess([k for k,t in client.operations].index('online_completed_1'),len(client.operations))

if __name__=='__main__':unittest.main()
