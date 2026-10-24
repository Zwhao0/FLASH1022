"""Explicit fixed-lifecycle startup heap policy; no per-step policy changes."""
import gc,json,os,platform,sys,time
from pathlib import Path
_OWNER=None
_RETAINED=False
POLICY=dict(schema='startup_process_lifetime_heap_freeze_v2',fixed_model_lifecycle=True,fixed_model_process_lifetime=True,preexisting_heap_policy='retain_all_frozen_until_process_exit',collect_once=True,freeze_once=True,normal_dynamic_gc=True)
def memory(torch):
    rss=None;kind='unavailable'
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024;kind='current_rss_proc_status';break
    except OSError:pass
    if rss is None:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);kind='peak_rss_resource'
    cuda={}
    for name in ('current_device','memory_allocated','memory_reserved'):
        method=getattr(torch.cuda,name,None)
        cuda[name]=int(method()) if method else None
    return dict(rss_bytes=rss,rss_kind=kind,cuda=cuda)
def state(torch):
    return dict(enabled=gc.isenabled(),thresholds=list(gc.get_threshold()),counts=list(gc.get_count()),frozen_objects=gc.get_freeze_count(),memory=memory(torch))
class StartupHeapFreeze:
    def __init__(self,torch,rank,plan):
        global _OWNER
        if plan.get('startup_gc_policy')!=POLICY:raise ValueError('explicit fixed-model startup GC opt-in required')
        if any(not hasattr(gc,name) for name in ('freeze','unfreeze','get_freeze_count')):raise RuntimeError('GC freeze API unavailable')
        if _OWNER is not None or _RETAINED or not gc.isenabled():raise RuntimeError('GC disabled or existing lifecycle owner; refusing takeover')
        self.torch,self.rank=torch,rank;self.owned=False;self.closed=False;self.preexisting=gc.get_freeze_count()
        self.record=dict(schema=POLICY['schema'],rank=rank,python_version=sys.version,implementation=platform.python_implementation(),policy=POLICY,before=state(torch),startup_before_any_batch=True)
        self.thresholds=gc.get_threshold();start=time.perf_counter()
        self.record['startup_collected']=gc.collect()
        gc.freeze();self.owned=True;_OWNER=self
        self.startup_seconds=time.perf_counter()-start
        try:
            self.record.update(startup_seconds=self.startup_seconds,after_startup=state(torch))
            self.record['incremental_frozen_objects']=self.record['after_startup']['frozen_objects']-self.preexisting
            self.record['preexisting_frozen_objects']=self.preexisting
            if not gc.isenabled() or gc.get_threshold()!=self.thresholds:raise RuntimeError('GC policy changed during startup')
        except BaseException:
            self.restore(None,'startup_failed');raise
    def restore(self,output,status):
        global _OWNER,_RETAINED
        if self.closed:return 0.0
        start=time.perf_counter();self.record['before_restore']=state(self.torch)
        if not self.owned or _OWNER is not self:raise RuntimeError('startup GC ownership lost; refusing unfreeze')
        if self.preexisting==0:gc.unfreeze()
        else:_RETAINED=True
        self.owned=False;_OWNER=None;self.closed=True
        self.record.update(status=status,after_restore=state(self.torch),restore_seconds=time.perf_counter()-start,
            policy_preserved=gc.isenabled() and gc.get_threshold()==self.thresholds,unfreeze_owned_heap_only=self.preexisting==0,preexisting_heap_preserved=self.preexisting>0,
            heap_restored=self.preexisting==0,exit_retention_reason='preexisting permanent generation cannot be selectively unfrozen; fixed model retained until training subprocess exit' if self.preexisting else None)
        if output is not None:
            Path(output,f'startup_gc_freeze_rank{self.rank}.json').write_text(json.dumps(self.record,indent=2,sort_keys=True)+'\n')
        return time.perf_counter()-start

# Explicit production launcher adapter; default plans never enter this path.
def configure_plan(plan):
    from flash_v1.stage_forward_sampling import CONFIG
    if (not plan.get('online') or not plan.get('evidence_enabled') or not plan.get('stage_forward_aux_enabled')
            or not plan.get('stage_forward_aux_model') or not plan.get('stage_forward_aux_context_binding')):
        raise ValueError('startup sparse GC requires complete explicit online TP-peer configuration')
    return dict(plan,stage_forward_aux_sampling=dict(CONFIG),startup_gc_fixed_model=True,startup_gc_policy=dict(POLICY))
def validate_plan(plan):
    flag=plan.get('startup_gc_fixed_model',False)
    if type(flag) is not bool:raise ValueError('startup_gc_fixed_model must be boolean')
    if not flag:
        if plan.get('startup_gc_policy') is not None:raise ValueError('startup GC policy requires explicit fixed-model flag')
        return False
    if (plan.get('startup_gc_policy')!=POLICY or plan.get('monitor_benchmark')
            or plan.get('stage_forward_aux_sampling')!=configure_plan(plan)['stage_forward_aux_sampling']):
        raise ValueError('invalid fixed-model startup GC contract')
    return True

def install(torch,rank,plan,output):
    import atexit,functools
    if not validate_plan(plan):return None
    owner=StartupHeapFreeze(torch,rank,plan)
    owner.exit_callback=functools.partial(owner.restore,output,'atexit')
    atexit.register(owner.exit_callback)
    return owner

def restore_owned(owner,output,status):
    import atexit
    if owner is None:return 0.0
    try:return owner.restore(output,status)
    finally:
        if owner.closed:atexit.unregister(owner.exit_callback)
