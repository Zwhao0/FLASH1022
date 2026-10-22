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
        if any(self.rank in row.get('target_ranks', []) for row in self.intervals):
            def measure(cycles):
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record(); torch.cuda._sleep(cycles); b.record(); b.synchronize()
                return a.elapsed_time(b)
            measure(2_000_000)
            ms = measure(2_000_000)
            if ms <= 0:
                raise ValueError('invalid CUDA delay calibration')
            self.cycles_per_ms = 2_000_000 / ms
            target = max(row.get('delay_ms', 0) for row in self.intervals)
            observed = measure(round(self.cycles_per_ms * target))
            if abs(observed - target) > max(0.5, target * .25):
                raise ValueError(f'CUDA delay calibration mismatch: {target}, {observed}')
            self.truth.write(json.dumps(dict(event='calibration', rank=self.rank,
                requested_ms=target, observed_ms=observed, cycles_per_ms=self.cycles_per_ms)) + '\n')

    def attach_backward(self, model):
        layers = model.model.layers
        attention = [layer.self_attn for layer in layers]
        experts = [expert for layer in layers if hasattr(layer.mlp, 'experts')
                   for expert in layer.mlp.experts if not isinstance(expert, torch.nn.Identity)]
        groups = [attention]
        if self.args.delay_placement == 'attention-experts':
            if not experts:
                raise ValueError('no owned experts for injection')
            groups.append(experts)
        def inject(fraction):
            if self.interval and self.rank in self.interval.get('target_ranks', []):
                cycles = round(self.cycles_per_ms * self.interval['delay_ms'] * fraction)
                if cycles:
                    torch.cuda._sleep(cycles)
                    self.calls += 1
                    self.cycles_injected += cycles
        for modules in groups:
            fraction = 1 / len(groups) / len(modules)
            for module in modules:
                self.hooks.append(module.register_full_backward_pre_hook(
                    lambda module, grad, fraction=fraction: inject(fraction)))

    def begin_iteration(self, iteration):
        self.iteration = iteration
        self.interval = next((row for row in self.intervals
                              if row['start'] <= iteration <= row['end']), None)
        if self.interval is None:
            raise ValueError(f'collection plan does not cover iteration {iteration}')
        self.context[:] = struct.pack('QQ', iteration, 0)
        self.calls = self.cycles_injected = 0

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
            self.truth.write(json.dumps(dict(rank=self.rank, iteration=iteration,
                hook_calls=self.calls, injected_cycles=self.cycles_injected,
                calibrated_injected_ms=self.cycles_injected / self.cycles_per_ms,
                kind='simulated_gpu_stream_delay')) + '\n')

    def close(self, iteration):
        set_collection_context(None)
        self.context[:] = struct.pack('QQ', 0, 0)
        self.context.close(); self.context_file.close()
        self.log.close(); self.truth.close()
        for hook in self.hooks:
            hook.remove()
