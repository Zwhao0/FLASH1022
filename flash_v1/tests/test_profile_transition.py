import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from flash_v1 import localize_v2_linear as linear
from flash_v1.profile_transition import candidate
from flash_v1.online import controller,profile_transition_event
from flash_v1.online_v2 import COMM_COUNTS,PHASE_CODES
from flash_v1.monitor_channel import MonitorChannel


def evidence(onset=1,recover=None,other=None):
    rows=[]
    for step in range(8):
        for rank in range(4):
            for (slot,phase),count in COMM_COUNTS.items():
                slow=(rank==0 and step>=onset and (recover is None or step<recover)) or (rank==other and step>=5)
                duration=400. if slow else 100.
                intervals=[dict(start_us=i*10,end_us=(i+1)*10,phase_code=256|PHASE_CODES[slot,phase]) for i in range(count)]
                rows.append(dict(record_type='measurement',rank=rank,iteration=step,epoch=1,slot=slot,phase=phase,
                    layer=0,dtype='torch.bfloat16',shape=[16,896],stream_id=0,measurement='local_residual',coverage_complete=True,
                    duration_us=duration,module_elapsed_us=duration+count*10,communication_intervals=intervals))
    return rows


class TransitionTests(unittest.TestCase):
    def setUp(self):
        healthy=evidence(onset=100)
        self.model=linear.fit(healthy,[healthy],fit_sessions=['fit'],calibration_sessions=['cal'])
        self.schedule=dict(epoch=1,start=0,end=7)
    def infer_candidate(self,rows):
        return candidate(rows,self.model,range(4),self.schedule,linear.infer(rows,self.model,range(4)))
    def test_only_observed_onset_with_persistent_unique_tail(self):
        self.assertEqual(self.infer_candidate(evidence())['rank'],0)
        self.assertIsNone(self.infer_candidate(evidence(onset=0)))
        self.assertIsNone(self.infer_candidate(evidence(onset=100)))
        self.assertIsNone(self.infer_candidate(evidence(recover=5)))
        self.assertIsNone(self.infer_candidate(evidence(other=1)))
    def test_unknown_or_incomplete_cannot_justify_continuation(self):
        rows=evidence();self.assertIsNone(self.infer_candidate(rows[:-1]))
        rows=evidence();rows[0]['shape']=[999,999];self.assertIsNone(self.infer_candidate(rows))
        rows=evidence();rows[0]['coverage_complete']=False;self.assertIsNone(self.infer_candidate(rows))
    def test_other_rank_unknown_is_skipped_without_changing_parent_infer(self):
        rows=evidence()
        for row in rows:
            if row['rank']==1 and row['slot']=='experts':row['shape']=[999,999]
        result=linear.infer(rows,self.model,range(4))
        self.assertEqual((result['status'],result['rank']),('localized',0))
        self.assertEqual(self.infer_candidate(rows)['rank'],0)
    def test_target_unknown_first_or_tail_still_rejected(self):
        for step in (0,5,6,7):
            rows=evidence()
            next(r for r in rows if r['rank']==0 and r['iteration']==step)['shape']=[999,999]
            self.assertIsNone(self.infer_candidate(rows))
    def test_invalid_non_target_metadata_is_not_unknown_reference(self):
        for field,values in [('shape',[None,'16',16,{},[-1],[True],[1.5]]),
                             ('dtype',[None,'','  ',16,[],{}])]:
            for value in values:
                with self.subTest(field=field,value=value):
                    rows=evidence();row=next(r for r in rows if r['rank']==1)
                    if value is None:row.pop(field)
                    else:row[field]=value
                    # The helper must fail closed before localizer schema parsing.
                    self.assertIsNone(candidate(rows,self.model,range(4),self.schedule,dict(status='localized',rank=0)))
    def test_scalar_shape_is_valid_unknown_non_target_metadata(self):
        rows=evidence();next(r for r in rows if r['rank']==1)['shape']=[]
        self.assertEqual(self.infer_candidate(rows)['rank'],0)
    def test_diagnose_opt_in_preserves_parent_prediction_and_requires_markers(self):
        from flash_v1.online_v2 import diagnose
        rows=evidence()
        markers=[dict(record_type='iteration_complete',epoch=1,rank=r,iteration=i,rows_count=4)
                 for r in range(4) for i in range(8)]
        class Client:
            def __init__(self,records):self.records=records
            def lrange(self,key,*args):return [json.dumps(r) for r in self.records] if key.endswith('0') else []
        with tempfile.TemporaryDirectory() as d:
            Path(d,'topology.json').write_text(json.dumps(dict(world_size=4)))
            parent=diagnose(Client(rows+markers),d,self.schedule,self.model)
            self.assertNotIn('profile_transition',parent)
            Path(d,'configuration.json').write_text(json.dumps(dict(monitor_profile_transition=True)))
            enabled=diagnose(Client(rows+markers),d,self.schedule,self.model)
            self.assertEqual(enabled['predicted'],parent['predicted'])
            self.assertEqual(enabled['profile_transition']['rank'],0)
            self.assertIsNone(diagnose(Client(rows+markers[:-1]),d,self.schedule,self.model))
    def test_child_quota_done_late_tail_and_rank_mismatch(self):
        cfg=dict(monitor_profile_transition=True,training_steps=100)
        pending=dict(schedule=dict(start=2,end=9));result=dict(status='localized',rank=0,profile_transition={'rank':0})
        event=profile_transition_event(cfg,pending,result,1,3,10,None)
        self.assertEqual((event['iteration'],event['epoch'],event['parent_epoch']),(10,2,1))
        for c,p,r,quota,done in [({},pending,result,3,None),(cfg,dict(pending,parent_epoch=1),result,3,None),
                (cfg,pending,result,1,None),(dict(cfg,training_steps=18),pending,result,3,None),
                (cfg,pending,result,3,{'last_iteration':10}),(cfg,pending,dict(result,rank=1),3,None)]:
            self.assertIsNone(profile_transition_event(c,p,r,1,quota,10,done))
    def test_real_child_request_ack_order_and_no_recursive_child(self):
        class Training:
            def __init__(self):self.values={};self.polls=0;self.operations=[]
            def exists(self,key):return key=='shutdown' and 'online_completed_2' in self.values
            def get(self,key):return self.values.get(key)
            def set(self,key,value):
                self.operations.append((key,value));self.values[key]=value
                if key=='online_request':
                    epoch=json.loads(value)['epoch']
                    if epoch==2:
                        assert 'online_completed_1' in self.values
                        assert self.values['online_pending']=='1'
                    start=2 if epoch==1 else 12
                    self.values[f'online_schedule_{epoch}']=json.dumps(dict(epoch=epoch,start=start,end=start+7))
            def lrange(self,key,start,end):
                if start==-1:return [json.dumps(dict(iteration=10 if self.polls<3 else 20))]
                self.polls+=1;last=1 if self.polls==1 else 10 if self.polls==2 else 20
                return [json.dumps(dict(iteration=i,wall_iteration_ms=100,profile_epoch=1 if 2<=i<=9 else 2 if 12<=i<=19 else 0)) for i in range(1,last+1)]
        class Detector:
            def __init__(self,warmup):pass
            def add(self,iteration,duration):return dict(iteration=iteration)
        client=Training();hint=dict(status='localized',rank=0,profile_transition={'rank':0})
        with tempfile.TemporaryDirectory() as output:
            Path(output,'configuration.json').write_text(json.dumps(dict(monitor_profile_transition=True,training_steps=100)))
            with patch('flash_v1.online.WindowDetector',Detector),patch('flash_v1.online.drains_after',return_value=True),patch('flash_v1.online.diagnose',return_value=hint),patch('flash_v1.online.time.sleep'):
                controller(client,output,{},max_profiles=4)
            events=[json.loads(x) for x in Path(output,'events.jsonl').read_text().splitlines()]
        detections=[e for e in events if e['event']=='detection']
        self.assertEqual(len(detections),2)
        self.assertEqual((detections[1]['iteration'],detections[1]['source']),(10,'profile_transition'))
        self.assertEqual([json.loads(v)['epoch'] for k,v in client.operations if k=='online_request'],[1,2])
        self.assertEqual(client.values['online_pending'],'0')
    def test_latest_mailbox_read_does_not_consume(self):
        with tempfile.TemporaryDirectory() as d:
            channel=MonitorChannel.create(Path(d,'monitor.bin'))
            self.assertEqual(channel.latest_completed_iteration(),0)
            channel.publish(1,100,0,1.);self.assertEqual(channel.latest_completed_iteration(),1)
            rows,cursor=channel.read_since(0);self.assertEqual((len(rows),cursor),(1,1));channel.close()

if __name__=='__main__':unittest.main()
