"""Integration-only fixed FWD hooks; no TorchDispatch or tensor modification."""
import math
from contextlib import contextmanager
from flash_v1.fault_workloads import StreamWorkload


class NativeModuleFMA:
    def __init__(self, torch, stage, rank, cases, units_per_ms, workload_factory=StreamWorkload):
        if not math.isfinite(units_per_ms) or units_per_ms<=0:
            raise ValueError('positive frozen FMA slope required')
        self.rank,self.cases,self.units_per_ms=rank,cases,units_per_ms
        self.iteration=0;self.counts={};self.stats=[];self.handles=[];self.engine=None
        self.factory,self.torch=workload_factory,torch
        modules=dict(stage.named_modules())
        for case in cases:
            if case['kind']=='module':
                if case['module_path'] not in modules:
                    raise ValueError('fixed native module absent')
                def observe(module,args,kwargs,output,case=case):
                    self._observe(case)
                self.handles.append(modules[case['module_path']].register_forward_hook(observe,with_kwargs=True))
            elif case['kind']!='second_microbatch_loss':
                raise ValueError('unknown native integration site')
        # Only the assigned GPU allocates private fault work; context identity is
        # based on the global dispatch observer, never this private target state.
        if any(case['rank']==rank for case in cases):
            self.engine=workload_factory(torch,'cuda_fp32_fma')
            self.engine.launch_units(4096)  # Fixed startup warmup, no measurements.

    def begin_iteration(self,iteration):
        self.iteration=iteration;self.counts={}
        self.current=dict(iteration=iteration,rank=self.rank,selected_calls=0,injected_launches=0,
                          requested_extra_ms=0.,cases=[])

    def _observe(self,case):
        ident=case['case_id'];self.counts[ident]=self.counts.get(ident,0)+1
        call=self.counts[ident]
        if not (case['start']<=self.iteration<=case['end']) or case['rank']!=self.rank:
            return
        if case['kind']=='second_microbatch_loss' and call!=2:
            return
        expected=1 if case['kind']=='second_microbatch_loss' else 2
        milliseconds=case['extra_ms']/expected
        units=max(1,round(milliseconds*self.units_per_ms))
        if units>2_147_483_647:
            raise ValueError('native FMA rounds exceed signed int')
        self.engine.launch_units(units)
        self.current['selected_calls']+=1;self.current['injected_launches']+=1
        self.current['requested_extra_ms']+=milliseconds
        self.current['cases'].append(dict(case_id=ident,call=call,units=units,requested_ms=milliseconds))

    @contextmanager
    def loss_wrapper(self,functional):
        original=functional.cross_entropy
        cases=[case for case in self.cases if case['kind']=='second_microbatch_loss']
        if not cases:
            yield;return
        def measured(*args,**kwargs):
            result=original(*args,**kwargs)
            for case in cases:self._observe(case)
            return result
        functional.cross_entropy=measured
        try:yield
        finally:
            if functional.cross_entropy is not measured:
                raise RuntimeError('native loss wrapper replaced')
            functional.cross_entropy=original

    def end_iteration(self):
        self.current['observed_site_calls']=dict(self.counts)
        self.stats.append(self.current)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
