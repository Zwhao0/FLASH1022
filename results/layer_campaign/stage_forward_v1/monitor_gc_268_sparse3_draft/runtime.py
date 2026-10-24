"""Isolated cost replay backend; imported only by generated cost training source."""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

from .contract import position
from flash_v1.startup_gc import install,restore_owned


def atomic(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True)+'\n');temporary.replace(path)


def tensor_bytes(torch, value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def digest(torch, value):
    result=hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            result.update(str((str(item.dtype),list(item.shape))).encode());result.update(tensor_bytes(torch,item))
        elif isinstance(item,dict):
            for key in sorted(item,key=str):result.update(str(key).encode());add(item[key])
        elif isinstance(item,(list,tuple)):
            for child in item:add(child)
        elif hasattr(item,'tobytes'):
            result.update(str(getattr(item,'dtype','')).encode());result.update(item.tobytes())
        else:result.update(pickle.dumps(item,protocol=4))
    add(value);return result.hexdigest()


def cpu_clone(torch, value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_clone(torch,item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_clone(torch,item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_clone(torch,item) for item in value)
    return copy.deepcopy(value)


class Checkpoint:
    def __init__(self,torch,stage,optimizer,generator):
        self.torch,self.stage,self.optimizer,self.generator=torch,stage,optimizer,generator
        self.parameters=cpu_clone(torch,dict(stage.named_parameters()))
        self.buffers=cpu_clone(torch,dict(stage.named_buffers()))
        self.optimizer_state=cpu_clone(torch,optimizer.state_dict())
        self.cpu_rng=torch.get_rng_state().clone();self.cuda_rng=torch.cuda.get_rng_state().clone()
        self.private_rng=generator.get_state().clone();self.python_rng=random.getstate()
        try:
            import numpy
            self.numpy=numpy;self.numpy_rng=copy.deepcopy(numpy.random.get_state())
        except ImportError:self.numpy=None;self.numpy_rng=None
        self.parameter_digest=digest(torch,self.parameters)
    def restore(self):
        with self.torch.no_grad():
            for name,parameter in self.stage.named_parameters():parameter.copy_(self.parameters[name])
            for name,buffer in self.stage.named_buffers():buffer.copy_(self.buffers[name])
        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        self.torch.set_rng_state(self.cpu_rng.clone());self.torch.cuda.set_rng_state(self.cuda_rng.clone())
        self.generator.set_state(self.private_rng.clone());random.setstate(self.python_rng)
        if self.numpy is not None:self.numpy.random.set_state(copy.deepcopy(self.numpy_rng))


class BareDelay:
    """No FLASH context/telemetry IO on the bare path; setup object only."""
    def begin_iteration(self,*args):pass
    def end_iteration(self,*args):pass


def hook_census(stage):
    return [(id(module),tuple(len(getattr(module,name,{})) for name in
        ('_forward_pre_hooks','_forward_hooks','_backward_pre_hooks','_backward_hooks')))
        for module in stage.modules()]


class CostReplay:
    def __init__(self,torch,dist,args,plan,output,rank,stage,optimizer,generator,evidence,stage_aux,channel,delay):
        if not plan.get('monitor_sameprocess_cost') or args.lr!=.001 or args.monitor_benchmark:
            raise ValueError('explicit real-SGD sameprocess contract required')
        if evidence is None or stage_aux is None:raise ValueError('full collectors required for B/C')
        self.torch,self.dist,self.args,self.plan=torch,dist,args,plan
        self.output,self.rank,self.stage,self.optimizer,self.generator=Path(output),rank,stage,optimizer,generator
        self.full_evidence,self.full_aux,self.channel=evidence,stage_aux,channel
        self.full_delay,self.bare_delay=delay,BareDelay()
        self.local_leader=int(os.environ.get('LOCAL_RANK',str(rank%8)))==0
        self.node=rank//8;self.sequence=0;self.last_mode=None;self.last_block=None;self.checkpoint=None
        self.rows=[];self.blocks=[];self.setup=[];self.inputs={};self.epoch=0;self.finished=False
        self.original_apis=[(stage_aux.dist,name,getattr(stage_aux.dist,name)) for name in ('all_reduce','all_to_all_single','all_gather')]
        self.original_apis.append((stage_aux.schedules,'forward_step',stage_aux.schedules.forward_step))
        from flash_v1 import tensor_parallel
        self.original_apis.append((tensor_parallel,'_all_reduce_with_phase',tensor_parallel._all_reduce_with_phase))
        self.original_hooks=hook_census(stage)
        self.initial_online=bool(args.online)
        self.assert_clean()
        self.original_optimizer_step=optimizer.step
        def measured_optimizer_step(*args,**kwargs):
            if getattr(self,'current',{}).get('block'):
                self.block_step_calls+=1
                self.block_lrs.extend(float(group['lr']) for group in optimizer.param_groups)
            return self.original_optimizer_step(*args,**kwargs)
        self.measured_optimizer_step=measured_optimizer_step
        optimizer.step=measured_optimizer_step
        self.heap_policy=install(torch,rank,plan,output)
        self.setup.append(dict(kind="startup_heap_collect_freeze",seconds=self.heap_policy.startup_seconds))
    def assert_clean(self):
        if (self.full_evidence.active or self.full_evidence.pending or self.full_evidence.handles
                or self.full_aux.active or self.full_aux.pending is not None
                or hook_census(self.stage)!=self.original_hooks
                or any(getattr(owner,name) is not original for owner,name,original in self.original_apis)):
            raise RuntimeError('active collector/hook/API at cost boundary')
        from flash_v1 import train
        if train._stage_forward_aux is not None:raise RuntimeError('active stage observer at cost boundary')
    def _boundary(self,mode,physical):
        self.sequence+=1;start=time.perf_counter();self.dist.barrier()
        request=self.output/f'cost_mode_request_node{self.node}.json';ack=self.output/f'cost_mode_ack_node{self.node}.json'
        if self.local_leader:
            item=position(physical) if 1<=physical<=268 else {}
            atomic(request,dict(sequence=self.sequence,mode=mode,physical=physical,block=item.get('block'),logical_mode=item.get('mode'),request_walltime=time.time()))
        deadline=time.monotonic()+30
        while True:
            value=json.loads(ack.read_text()) if ack.exists() else {}
            if value.get('sequence')==self.sequence:
                if value.get('status')!='ready' or value.get('mode')!=mode:raise RuntimeError('cost supervisor rejected boundary: '+str(value))
                break
            if time.monotonic()>deadline:raise TimeoutError('cost actor boundary handshake timed out')
            time.sleep(.01)
        self.dist.barrier();self.setup.append(dict(kind='actor_handshake',mode=mode,physical=physical,seconds=time.perf_counter()-start))
    def enter(self,physical):
        item=position(physical);switched=item['block']!=self.last_block
        if switched:
            self.assert_clean();start=time.perf_counter()
            if physical==13:
                self.torch.cuda.synchronize();self.checkpoint=Checkpoint(self.torch,self.stage,self.optimizer,self.generator)
                self.setup.append(dict(kind='checkpoint',physical=physical,seconds=time.perf_counter()-start));start=time.perf_counter()
            if item['block']:
                if self.checkpoint is None:raise RuntimeError('missing post-warmup checkpoint')
                self.checkpoint.restore();self.torch.cuda.synchronize();self.dist.barrier()
                self.inputs={};self.block_rows=[]
                self.block_step_calls=0;self.block_lrs=[]
                self.setup.append(dict(kind='restore',block=item['block'],seconds=time.perf_counter()-start))
            self.last_block=item['block']
        enabled=item['mode']!='A'
        if switched and physical!=13:
            self._boundary('full' if enabled else 'bare',physical)
        self.last_mode=item['mode'];self.current=item
        self.args.online=enabled
        self.selected_collector=enabled
        return enabled,(self.full_evidence if enabled else None),(self.full_aux if enabled else None),(self.full_delay if enabled else self.bare_delay),switched
    def begin(self,physical):
        if self.current['block'] and self.current['logical'] ==33:
            self.torch.cuda.synchronize()
            self.setup.append(dict(kind='window_start_fence',physical=physical,logical=self.current['logical'],outside_window=True))
        self.step_started=time.perf_counter()
    def remember_inputs(self,physical,batches):
        # Retain references only: no D2H/hash/extra GPU work inside the timed step.
        if self.current['block']:self.inputs[self.current['logical']]=batches
    def note_main_drain(self,physical,evidence,auxiliary):
        self.main_drain_before=None
        if self.current['block'] and self.current['logical'] ==64:
            self.main_drain_before=dict(role_rows=len(evidence.pending) if evidence is not None else 0,
                                       aux_pending=auxiliary is not None and auxiliary.pending is not None)
    def end(self,physical,loss,profile_epoch):
        item=self.current
        expected_epoch=0
        if profile_epoch!=expected_epoch:raise RuntimeError('actual PROFILE does not match exact C41..48 contract')
        boundary=None
        if item['block'] and item['logical'] ==64:
            def pending():return dict(role_rows=len(self.full_evidence.pending),aux_pending=self.full_aux.pending is not None)
            before=pending();self.torch.cuda.synchronize()
            if item['mode']!='A':
                if self.full_evidence.pending:self.full_evidence.drain()
                if self.full_aux.pending is not None:
                    auxiliary=self.full_aux.drain()
                    if auxiliary is not None:
                        with (self.output/f'stage_forward_aux_rank{self.rank}.jsonl').open('a') as stream:
                            stream.write(json.dumps(auxiliary)+'\n')
            after=pending()
            if after['role_rows'] or after['aux_pending']:raise RuntimeError('window end leaves deferred collection')
            boundary=dict(before_pending=before,after_pending=after,gpu_fence_before_end_timestamp=True,
                          before_normal_main_drain=self.main_drain_before,
                          main_thread_drain_serialize_write_in_window=True)
        completed=time.perf_counter()
        row=dict(**item,loss=loss,full_step_wall_ms=(completed-self.step_started)*1000,
                 actual_profile_epoch=profile_epoch,
                 actual_online=bool(self.args.online),collector_references_present=self.selected_collector,
                 collection_context_bypassed=item['mode']=='A',
                 finished_walltime=time.time())
        if boundary is not None:row['window_end']=boundary
        self.rows.append(row)
        if item['block']:self.block_rows.append(row)
        if item['block'] and item['logical']==64:
            self.assert_clean();start=time.perf_counter()
            input_summary={str(logical):dict(content_sha256=digest(self.torch,batches),
                microbatches=[{key:dict(shape=list(tensor.shape),dtype=str(tensor.dtype),numel=tensor.numel()) for key,tensor in mb.items()} for mb in batches])
                for logical,batches in sorted(self.inputs.items())}
            parameters=digest(self.torch,dict(self.stage.named_parameters()))
            if parameters==self.checkpoint.parameter_digest:raise RuntimeError('no actual SGD parameter update')
            block=dict(block=item['block'],mode=item['mode'],parameter_sha256=parameters,
                       initial_parameter_sha256=self.checkpoint.parameter_digest,
                       optimizer_step_calls=self.block_step_calls,
                       observed_lr_min=min(self.block_lrs) if self.block_lrs else None,
                       observed_lr_max=max(self.block_lrs) if self.block_lrs else None,
                       collector_hooks_empty_and_APIs_original=True,
                       buffer_sha256=digest(self.torch,dict(self.stage.named_buffers())),
                       optimizer_sha256=digest(self.torch,self.optimizer.state_dict()),input_summary=input_summary,
                       private_rng_sha256=digest(self.torch,self.generator.get_state()),
                       cpu_rng_sha256=digest(self.torch,self.torch.get_rng_state()),cuda_rng_sha256=digest(self.torch,self.torch.cuda.get_rng_state()),
                       losses=[row['loss'] for row in self.block_rows])
            if block['optimizer_step_calls']!=64 or block['observed_lr_min']!=.001 or block['observed_lr_max']!=.001:
                raise RuntimeError('block did not execute64 real positive-LR SGD steps')
            if self.blocks:
                reference=self.blocks[0]
                for key in ('parameter_sha256','buffer_sha256','optimizer_sha256','input_summary','private_rng_sha256','cpu_rng_sha256','cuda_rng_sha256'):
                    if block[key]!=reference[key]:raise RuntimeError('replay block mismatch: '+key)
                if self.rank==0 and max(abs(a-b) for a,b in zip(block['losses'],reference['losses']))>=1e-4:
                    raise RuntimeError('logical-position loss trajectory mismatch')
            self.blocks.append(block);self.inputs={}
            self.setup.append(dict(kind='block_checks',block=item['block'],seconds=time.perf_counter()-start))
            self._save()
    def _save(self):
        atomic(self.output/f'cost_replay_rank{self.rank}.json',dict(status='complete' if self.finished else 'partial',
            source='isolated_test_only',rows=self.rows,blocks=self.blocks,setup=self.setup,
            actual_lr=self.args.lr,no_input_generation_cache=True,parameters_replayed_exact=True if len(self.blocks)==4 else None))
    def finish(self):
        if self.finished:return
        self.assert_clean()
        if len(self.blocks)!=4:raise RuntimeError('incomplete four replay blocks')
        self._boundary('done',269)
        self.setup.append(dict(kind='startup_heap_restore',seconds=restore_owned(self.heap_policy,self.output,'complete')))
        self.finished=True;self._save()
        self.args.online=self.initial_online
        self.optimizer.step=self.original_optimizer_step
        return self.full_delay
    def abort(self):
        # No distributed barrier in a failing rank: supervisors always resume
        # owned actors in their finally blocks. A local emergency request helps
        # normal peers terminate without deadlock; it is never a cost sample.
        if not self.finished and self.local_leader:
            self.sequence+=1;atomic(self.output/f'cost_mode_request_node{self.node}.json',dict(sequence=self.sequence,mode='abort',physical=0))
        restore_owned(self.heap_policy,self.output,'aborted')
        self._save()
        self.optimizer.step=self.original_optimizer_step
