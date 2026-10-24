"""CPU fake tensor/SGD/RNG replay and real-channel/actor boundary tests. No torch import."""
import copy
import json
from pathlib import Path
import random
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
import importlib.util
_prod=HERE.parent/'sparse3_gc_production_draft/production/flash_v1'
for _name in ('stage_forward_sampling','startup_gc'):
    _spec=importlib.util.spec_from_file_location('flash_v1.'+_name,_prod/(_name+'.py'));_module=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_module;_spec.loader.exec_module(_module)
from results.layer_campaign.stage_forward_v1.monitor_gc_268_sparse3_draft import runtime,node_cost
_actual_install=runtime.install
def _install_fixture(torch,rank,plan,output):
    from flash_v1.startup_gc import configure_plan
    full=dict(plan,online=True,evidence_enabled=True,stage_forward_aux_enabled=True,stage_forward_aux_model='model',stage_forward_aux_context_binding='binding')
    return _actual_install(torch,rank,configure_plan(full),output)
runtime.install=_install_fixture
from results.layer_campaign.stage_forward_v1.monitor_gc_268_sparse3_draft.contract import position
from flash_v1.monitor_channel import MonitorChannel

EVENTS=[]
class Tensor:
    dtype='fake.float';shape=(2,)
    def __init__(self,values):self.values=list(values);self.shape=(len(values),)
    def detach(self):return self
    def cpu(self):EVENTS.append('cpu_read');return self
    def contiguous(self):return self
    def view(self,dtype):return self
    def numpy(self):return SimpleNamespace(tobytes=lambda:json.dumps(self.values).encode())
    def clone(self):return Tensor(self.values)
    def copy_(self,other):self.values=list(other.values);return self
    def numel(self):return len(self.values)
class Generator:
    def __init__(self):self.value=7
    def get_state(self):return Tensor([self.value])
    def set_state(self,t):self.value=t.values[0]
    def next(self):self.value=(17*self.value+3)%997;return self.value
class CUDA:
    def __init__(self):self.state=Tensor([4])
    def get_rng_state(self):return self.state.clone()
    def set_rng_state(self,state):self.state=state.clone()
    def synchronize(self):EVENTS.append('fence')
class Torch:
    uint8='fake.uint8'
    def __init__(self):self.state=Tensor([3]);self.cuda=CUDA()
    def is_tensor(self,value):return isinstance(value,Tensor)
    def get_rng_state(self):return self.state.clone()
    def set_rng_state(self,state):self.state=state.clone()
    def no_grad(self):return __import__('contextlib').nullcontext()
class Stage:
    def __init__(self):self.weight=Tensor([1.,2.]);self.buffer=Tensor([0.])
    def named_parameters(self):return [('weight',self.weight)]
    def named_buffers(self):return [('buffer',self.buffer)]
    def modules(self):return [self]
class Optimizer:
    def __init__(self,stage):self.stage=stage;self.calls=0;self.physical_calls=0;self.grad=1.;self.param_groups=[dict(lr=.001)]
    def step(self):
        self.calls+=1;self.physical_calls+=1
        self.stage.weight.values=[x-.001*self.grad for x in self.stage.weight.values]
    def state_dict(self):return dict(calls=self.calls,param_groups=copy.deepcopy(self.param_groups))
    def load_state_dict(self,value):self.calls=value['calls'];self.param_groups=copy.deepcopy(value['param_groups'])
class Evidence:
    active=False
    def __init__(self):self.pending=[];self.handles=[]
    def drain(self):self.pending=[]
class Auxiliary:
    active=False
    def __init__(self):
        self.pending=None;self.dist=SimpleNamespace(all_reduce=lambda:None,all_to_all_single=lambda:None,all_gather=lambda:None)
        self.schedules=SimpleNamespace(forward_step=lambda:None)
    def drain(self):self.pending=None;return None
class Delay:
    def __init__(self):self.begin=0;self.end=0
    def begin_iteration(self,*args):self.begin+=1
    def end_iteration(self,*args):self.end+=1


class BackendTests(unittest.TestCase):
    def test_checkpoint_all_state_exact_and_python_rng(self):
        torch=Torch();stage=Stage();optimizer=Optimizer(stage);generator=Generator()
        checkpoint=runtime.Checkpoint(torch,stage,optimizer,generator)
        expected=random.random();checkpoint.restore();self.assertEqual(random.random(),expected)
        stage.weight.values=[9,10];stage.buffer.values=[8];optimizer.calls=88;generator.value=99
        torch.state=Tensor([98]);torch.cuda.state=Tensor([97]);checkpoint.restore()
        self.assertEqual(stage.weight.values,[1,2]);self.assertEqual(stage.buffer.values,[0])
        self.assertEqual((optimizer.calls,generator.value,torch.state.values,torch.cuda.state.values),(0,7,[3],[4]))
    def test_full268_replay_actual_sgd_input_rng_and_timer_boundaries(self):
        torch=Torch();stage=Stage();optimizer=Optimizer(stage);generator=Generator();evidence=Evidence();aux=Auxiliary();delay=Delay()
        fake_train=SimpleNamespace(_stage_forward_aux=None);fake_tp=SimpleNamespace(_all_reduce_with_phase=lambda:None)
        args=SimpleNamespace(lr=.001,monitor_benchmark=False,online=True)
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'flash_v1.train':fake_train,'flash_v1.tensor_parallel':fake_tp}):
            import flash_v1
            with patch.object(flash_v1,'train',fake_train,create=True),patch.object(flash_v1,'tensor_parallel',fake_tp,create=True):
                replay=runtime.CostReplay(torch,SimpleNamespace(barrier=lambda:None),args,{'monitor_sameprocess_cost':True},folder,0,stage,optimizer,generator,evidence,aux,SimpleNamespace(get_request=lambda:1 if physical<205 else 2),delay)
                requests=[];replay._boundary=lambda mode,physical:requests.append((mode,physical))
                # Use a real counter clock to spy fence vs timer/data-read order.
                real_clock=runtime.time.perf_counter
                def clock():EVENTS.append('clock');return real_clock()
                with patch.object(runtime.time,'perf_counter',side_effect=clock):
                    for physical in range(1,269):
                        enabled,e,a,d,switched=replay.enter(physical)
                        p=position(physical)
                        if p['mode']=='C' and p['logical']==40:
                            pass
                        EVENTS.clear();replay.begin(physical)
                        if p['logical'] in (33,) and p['block']:
                            self.assertEqual(EVENTS[:2],['fence','clock'])
                        batches=[dict(input_ids=Tensor([generator.next(),generator.next()]),next_token_labels=Tensor([generator.next()])) for _ in range(2)]
                        replay.remember_inputs(physical,batches)
                        d.begin_iteration(physical);optimizer.grad=sum(batches[0]['input_ids'].values)/100
                        optimizer.step();stage.buffer.values[0]+=1
                        torch.state.values[0]+=1;torch.cuda.state.values[0]+=1
                        d.end_iteration(physical,0)
                        if p['profile']:
                            evidence.pending=[1]*4;aux.pending={'fake':True}
                        replay.note_main_drain(physical,e,a)
                        if e is not None:e.drain()
                        if a is not None:a.drain()
                        epoch=(1 if p['block']==3 else 2) if p['profile'] else 0
                        loss=sum(stage.weight.values)+batches[0]['input_ids'].values[0]
                        if p['mode']=='C' and p['logical']==40:
                            # enter's request test needs the appropriate epoch.
                            pass
                        end_start=len(EVENTS);replay.end(physical,loss,epoch)
                        if p['logical']==64 and p['block']:
                            tail=EVENTS[end_start:]
                            self.assertEqual(tail[:2],['fence','clock']);self.assertGreater(tail.index('cpu_read'),1)
                        else:self.assertNotIn('cpu_read',EVENTS)
                replay.finish()
                self.assertEqual(optimizer.physical_calls,268)
                self.assertEqual((delay.begin,delay.end),(128,128))
                self.assertEqual([b['optimizer_step_calls'] for b in replay.blocks],[64]*4)
                self.assertTrue(all(b['observed_lr_min']==b['observed_lr_max']==.001 for b in replay.blocks))
                self.assertEqual(len({b['parameter_sha256'] for b in replay.blocks}),1)
                self.assertTrue(all(b['parameter_sha256']!=b['initial_parameter_sha256'] for b in replay.blocks))
                self.assertEqual(requests,[('bare',1),('full',77),('full',141),('bare',205),('done',269)])
                self.assertEqual(json.loads((Path(folder)/'cost_replay_rank0.json').read_text())['status'],'complete')
                self.assertEqual(optimizer.step,replay.original_optimizer_step)
    def test_actor_all_boundaries_real_channel_and_idle_publisher(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);channel=MonitorChannel.create(output/'monitor.bin')
            states={91:'R'};signals=[];published=[];processed=[0]
            def kill(pid,sig):signals.append((pid,sig));states[pid]='T' if sig==signal.SIGSTOP else 'R'
            supervisor=node_cost.Supervisor(output,0,lambda publisher:published.append(1) or 0,kill=kill)
            supervisor._state=lambda process:states[process.pid]
            supervisor.controllers=[SimpleNamespace(pid=91,poll=lambda:None)]
            client=SimpleNamespace(get=lambda key:0 if key=='online_pending' else processed[0])
            publishers=[SimpleNamespace(partial={},client=client),SimpleNamespace(partial={},client=client)]
            supervisor.publishers=publishers
            boundaries=[('bare',1),('full',77),('full',141),('bare',205),('done',269)]
            last=77
            for sequence,(mode,physical) in enumerate(boundaries,1):
                if physical>77 and physical<=205:
                    for step in range(last,physical):channel.publish(step,100.,int(position(step)['profile']),float(step))
                    channel.read_since(0);processed[0]=physical-1;last=physical
                runtime.atomic(output/'cost_mode_request_node0.json',dict(sequence=sequence,mode=mode,physical=physical,request_walltime=1.))
                supervisor.poll()
                ack=json.loads((output/'cost_mode_ack_node0.json').read_text());self.assertEqual(ack['status'],'ready')
                if mode=='bare':
                    before=len(published);supervisor.publish(publishers[0]);self.assertEqual(len(published),before)
                    self.assertTrue(ack['controller_paused'] and ack['publisher_idle'])
            self.assertTrue(all(pid==91 for pid,sig in signals)) # Redis is never stopped.
            self.assertEqual(states[91],'R');self.assertEqual(supervisor.sequence,5);channel.close()
    def test_no_input_hash_in_remember_and_no_hidden_per_step_gpu_reads(self):
        import ast,inspect,textwrap
        node=ast.parse(textwrap.dedent(inspect.getsource(runtime.CostReplay.remember_inputs)))
        self.assertFalse(any(isinstance(x,ast.Call) for x in ast.walk(node)))
        source=(HERE/'train_pipeline_cost.py').read_text();tree=ast.parse(source)
        loop=next(x for x in ast.walk(tree) if isinstance(x,ast.For) and getattr(x.target,'id',None)=='iteration')
        body=ast.unparse(loop)
        self.assertLess(body.index('cost.begin(iteration)'),body.index('torch.randint'))
        self.assertGreater(body.index('cost.end(iteration'),body.index('print('))
        self.assertIn('cost.remember_inputs(iteration, batches)',body)
        self.assertIn('cost.note_main_drain(iteration, evidence, active_stage_aux)',body)


if __name__=='__main__':unittest.main()
