"""Episode acquisition with workload phases; fault truth stays outside probe rows."""
import json
import mmap
import os
from pathlib import Path
import struct
import time

import torch
import torch.distributed as dist

from flash_v1.tensor_parallel import set_collection_context
from flash_v1.fault_workloads import MODES, StreamWorkload


def add_arguments(parser):
    parser.add_argument('--collection-plan')
    parser.add_argument('--seed', type=int, default=2026)


class CollectionSession:
    def __init__(self, args, output):
        self.args, self.output = args, Path(output)
        self.rank = dist.get_rank()
        self.plan = json.loads(Path(args.collection_plan).read_text())
        self.intervals = self.plan['intervals']
        self.interval = None
        self.iteration = 0
        self.hooks = []
        self.calls = 0
        self.cycles_injected = 0
        self.cycles_per_ms = 0
        self.estimated_ms = 0
        self.engines = {}
        context = Path(os.environ['FLASH_CONTEXT_DIR']) / f'{self.rank}.bin'
        self.context_file = context.open('r+b')
        if os.fstat(self.context_file.fileno()).st_size != 16:
            raise ValueError('Collection context requires two uint64 fields (16 bytes)')
        self.context = mmap.mmap(self.context_file.fileno(), 16)
        self.profile_enabled = False
        self.set_profile_enabled(not self.plan.get('online', False)
                                 and self.plan.get('profile_enabled', True)
                                 and self.plan.get('probe_enabled', True))
        self.log = (self.output / f'telemetry_rank{self.rank}.jsonl').open('w', buffering=1)
        self.truth = (self.output / f'injection_rank{self.rank}.jsonl').open('w', buffering=1)
        for row in self.intervals:
            if row.get('fault_mode') is not None and row['fault_mode'] not in MODES:
                raise ValueError(f"unknown fault_mode: {row['fault_mode']}")

    def attach_backward(self, model):
        layers = model.model.layers
        attention = [layer.self_attn for layer in layers]
        experts = [expert for layer in layers if hasattr(layer.mlp, 'experts')
                   for expert in layer.mlp.experts if not isinstance(expert, torch.nn.Identity)]
        targeted = [row for row in self.intervals
                    if self.rank in row.get('target_ranks', []) and row.get('delay_ms', 0) > 0
                    and row.get('fault_mode') != 'whole_compute']
        selections = {}
        for row in targeted:
            mode = row.get('fault_mode')
            if mode is None:
                groups = [attention]
                if self.args.delay_placement == 'attention-experts':
                    groups.append(experts)
                phase, engine_name = 'backward', 'cuda_sleep'
            else:
                groups = [experts if mode == 'experts_bwd_sleep' else attention]
                phase = 'forward' if mode == 'attention_fwd_sleep' else 'backward'
                engine_name = 'cuda_fp32_fma' if mode.endswith('_compute') else 'cuda_sleep'
            if any(not modules for modules in groups):
                raise ValueError(f'no owned modules for injection: {mode}')
            engine = self.engines.get(engine_name)
            if engine is None:
                engine = self.engines[engine_name] = StreamWorkload(torch, engine_name)
            selection = []
            for modules in groups:
                fraction = 1 / len(groups) / len(modules)
                dose = row['delay_ms'] * fraction
                result = engine.prepare(dose)
                if engine_name == 'cuda_sleep':
                    self.cycles_per_ms = result['cycles_per_ms']
                self.truth.write(json.dumps(dict(event='calibration', rank=self.rank,
                    fault_mode=mode or 'legacy_attention_experts', **result)) + '\n')
                for module in modules:
                    selection.append((module, phase, fraction))
            selections[mode] = (engine_name, selection)
        # Each phase/module hook is installed once, even for repeated modes/strengths.
        hooks = {}
        for mode, (engine_name, selection) in selections.items():
            for module, phase, fraction in selection:
                hooks.setdefault((id(module), phase), (module, {}))[1][mode] = (engine_name, fraction)
        for (_, phase), (module, choices) in hooks.items():
            def inject(module, inputs, choices=choices):
                row = self.interval
                if not row or self.rank not in row.get('target_ranks', []):
                    return
                mode = row.get('fault_mode')
                if mode not in choices or row.get('delay_ms', 0) <= 0:
                    return
                engine_name, fraction = choices[mode]
                dose = row['delay_ms'] * fraction
                result = self.engines[engine_name].launch(dose)
                self.calls += 1
                self.estimated_ms += result['observed_ms']
                if engine_name == 'cuda_sleep':
                    self.cycles_injected += result['units']
            register = (module.register_forward_pre_hook if phase == 'forward'
                        else module.register_full_backward_pre_hook)
            self.hooks.append(register(inject))

    def begin_iteration(self, iteration):
        self.iteration = iteration
        self.interval = next((row for row in self.intervals
                              if row['start'] <= iteration <= row['end']), None)
        if self.interval is None:
            raise ValueError(f'collection plan does not cover iteration {iteration}')
        self.context[:] = struct.pack('QQ', iteration, 0)
        self.calls = self.cycles_injected = 0
        self.estimated_ms = 0

    def set_profile_enabled(self, enabled):
        self.profile_enabled = bool(enabled)
        # Phase labels are needed only inside a complete PROFILE iteration.
        # The injection plan and per-iteration timing are independent of this.
        set_collection_context(self.context if self.profile_enabled else None)

    def end_iteration(self, iteration, milliseconds):
        # Host monotonic timestamps are node-local; never compare them cross-node.
        if self.profile_enabled or self.plan.get('audit_telemetry', False):
            self.log.write(json.dumps(dict(rank=self.rank, iteration=iteration,
                time_ms=milliseconds, monotonic_us=time.monotonic_ns() // 1000)) + '\n')
        if self.calls:
            mode = self.interval.get('fault_mode')
            engine = 'cuda_fp32_fma' if mode == 'attention_bwd_compute' else 'cuda_sleep'
            self.truth.write(json.dumps(dict(rank=self.rank, iteration=iteration,
                hook_calls=self.calls, injected_cycles=self.cycles_injected,
                calibrated_injected_ms=self.estimated_ms, estimated_gpu_ms=self.estimated_ms,
                fault_mode=mode or 'legacy_attention_experts', engine=engine,
                kind=('simulated_gpu_compute' if engine == 'cuda_fp32_fma'
                      else 'simulated_gpu_stream_delay'))) + '\n')

    def close(self, iteration):
        set_collection_context(None)
        self.context[:] = struct.pack('QQ', 0, 0)
        self.context.close(); self.context_file.close()
        self.log.close(); self.truth.close()
        for hook in self.hooks:
            hook.remove()
