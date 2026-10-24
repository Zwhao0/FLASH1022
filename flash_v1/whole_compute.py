"""Healthy aten timing and proportional local-compute slowdown, with no tensor edits.

Enter this context around BOTH forward and backward. Warm up outside calibration;
collect several healthy iterations, finish_calibration(), then begin_iteration().
No events, tensor allocations or synchronization occur in injection dispatch.
"""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from flash_v1.fault_workloads import StreamWorkload


# Explicit local compute allowlist. c10d/custom namespaces never qualify.
COMPUTE_OPS = frozenset('''mm addmm bmm baddbmm mv dot matmul linear
_scaled_dot_product_flash_attention _scaled_dot_product_flash_attention_backward
_scaled_dot_product_efficient_attention _scaled_dot_product_efficient_attention_backward
_scaled_dot_product_cudnn_attention _scaled_dot_product_cudnn_attention_backward
_scaled_dot_product_attention_math _flash_attention_forward _flash_attention_backward
native_layer_norm native_layer_norm_backward native_group_norm native_group_norm_backward
layer_norm rms_norm _softmax _softmax_backward_data _log_softmax _log_softmax_backward_data
softmax log_softmax sum mean var std var_mean amax amin max min norm linalg_vector_norm
add sub mul div pow rsqrt sqrt reciprocal exp log neg abs square
silu silu_backward gelu gelu_backward relu threshold_backward sigmoid sigmoid_backward
tanh tanh_backward clamp clamp_min clamp_max where eq ne lt le gt ge
add_ sub_ mul_ div_ zero_ fill_ masked_fill masked_fill_ topk sort cumsum
scatter_add scatter_add_ index_add index_add_ index_put index_put_
embedding_dense_backward nll_loss_forward nll_loss_backward
nll_loss2d_forward nll_loss2d_backward _prelu_kernel _prelu_kernel_backward'''.split())


def _tensors(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype') and hasattr(value, 'device'):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _tensors(v)]
    return []


def _signature(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype') and hasattr(value, 'device'):
        return ('tensor', tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, (tuple, list)):
        return tuple(_signature(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _signature(v)) for k, v in sorted(value.items()))
    # Scalars affect dispatch (axes, alpha, causal, dropout); retain their values.
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def work_units(name, tensors):
    """Fallback estimate only; measured shape timings always take precedence."""
    shapes = [tuple(t.shape) for t in tensors]
    if name in ('mm', 'addmm', 'bmm', 'baddbmm', 'matmul', 'linear'):
        matrices = [s for s in shapes if len(s) >= 2]
        if len(matrices) >= 2:
            a, b = matrices[-2:]
            if name == 'linear':
                return max(1, 2 * math.prod(a) * b[0])
            return max(1, 2 * math.prod(a) * b[-1])
    if 'attention' in name and len(shapes) >= 3:
        # Backward starts with dO, then Q/K/V; skip dO in that case.
        offset = 1 if 'backward' in name else 0
        if len(shapes) > offset + 2:
            q, k, v = shapes[offset:offset + 3]
            if len(q) >= 3 and len(k) >= 3:
                cost = 2 * math.prod(q[:-2]) * q[-2] * k[-2] * (q[-1] + v[-1])
                return max(1, cost * (2.5 if 'backward' in name else 1))
    return max(1, sum(math.prod(s) for s in shapes))


class WholeComputeSlowdown:
    def __init__(self, torch, rank=0, samples_per_shape=3, quantum_ms=.05,
                 workload_factory=StreamWorkload):
        if samples_per_shape < 1 or quantum_ms <= 0:
            raise ValueError('positive samples_per_shape and quantum_ms required')
        self.torch, self.rank = torch, rank
        self.samples_per_shape, self.quantum_ms = samples_per_shape, quantum_ms
        self.workload_factory = workload_factory
        self.calibrating, self.ready = False, False
        self.samples, self.table, self.pending = defaultdict(list), {}, defaultdict(float)
        self.fallbacks, self.ignored = {}, Counter()
        self.engine = None
        self.mode = None
        self.begin_iteration(0)

    def __enter__(self):
        # Lazy import keeps metadata and CPU fake tests independent of torch.
        from torch.utils._python_dispatch import TorchDispatchMode
        owner = self
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return owner.dispatch(func, args, kwargs or {})
        self.mode = Mode()
        self.mode.__enter__()
        return self

    def __exit__(self, *args):
        return self.mode.__exit__(*args)

    def begin_calibration(self):
        if self.ready:
            raise RuntimeError('healthy calibration is already frozen')
        self.calibrating = True

    def begin_iteration(self, iteration, extra_ratio=0):
        if extra_ratio < 0 or not math.isfinite(extra_ratio):
            raise ValueError('slowdown extra ratio must be finite and nonnegative')
        if extra_ratio and not self.ready:
            raise RuntimeError('finish healthy calibration before injecting')
        self.iteration, self.extra_ratio = iteration, extra_ratio
        self.stats = dict(rank=self.rank, iteration=iteration, extra_ratio=extra_ratio,
                          pending_before_ms=sum(self.pending.values()),
                          compute_calls=0, injected_launches=0, healthy_estimated_ms=0.,
                          requested_extra_ms=0., calibrated_injected_ms=0., fallback_calls=0)
        self.iteration_fallbacks = Counter()
        self.op_stats = {}
        self.bucket_stats = {}

    def dispatch(self, func, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.calibrating and not self.extra_ratio:
            return func(*args, **kwargs)
        full_name = str(func)
        parts = full_name.split('.')
        name = parts[1] if len(parts) > 1 else full_name
        tensors = _tensors((args, kwargs))
        cuda = any(getattr(t.device, 'type', str(t.device)) == 'cuda' for t in tensors)
        selected = parts[0] == 'aten' and name in COMPUTE_OPS and cuda
        if not selected:
            if self.calibrating and cuda:
                self.ignored[full_name] += 1
            return func(*args, **kwargs)
        phase = ('bwd' if self.torch._C._current_graph_task_id() >= 0 else 'fwd')
        key = repr((phase, full_name, _signature(args), _signature(kwargs)))
        units = work_units(name, tensors)
        if self.calibrating:
            if len(self.samples[key]) < self.samples_per_shape:
                start, end = [self.torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                result = func(*args, **kwargs)
                end.record()
                self.samples[key].append((start, end, full_name, units, phase))
                return result
            return func(*args, **kwargs)
        result = func(*args, **kwargs)
        if self.extra_ratio:
            self._inject(key, full_name, units, phase)
        return result

    def finish_calibration(self, path=None):
        self.calibrating = False
        self.torch.cuda.synchronize()  # Healthy startup only.
        for key, samples in self.samples.items():
            times = [start.elapsed_time(end) for start, end, _, _, _ in samples]
            duration = median(times)
            if duration <= 0:
                raise RuntimeError('nonpositive healthy CUDA timing')
            self.table[key] = dict(op=samples[0][2], phase=samples[0][4], work_units=samples[0][3],
                                   median_ms=duration, samples_ms=times)
        if not self.table:
            raise RuntimeError('no CUDA compute operators observed during healthy calibration')
        self.catalog_sha256 = hashlib.sha256(json.dumps(self.table, sort_keys=True).encode()).hexdigest()
        self.samples.clear()  # Release CUDA event handles before measured training.
        return self._prepare_workload(path)

    def load_calibration(self, path, output_path=None):
        """Reuse fixed healthy shape medians; calibrate only this GPU's FMA kernel."""
        if self.ready or self.calibrating:
            raise RuntimeError('load reference before calibration or injection')
        reference = json.loads(Path(path).read_text())
        if reference['rank'] != self.rank:
            raise ValueError('reference rank mismatch')
        self.table = reference['shapes']
        actual_hash = hashlib.sha256(json.dumps(self.table, sort_keys=True).encode()).hexdigest()
        if reference['catalog_sha256'] != actual_hash:
            raise ValueError('reference catalog hash mismatch')
        self.catalog_sha256 = actual_hash
        self.reference_catalog_sha256 = actual_hash
        self.reference_path = str(path)
        self.ignored.update(reference.get('excluded_cuda_ops', {}))
        return self._prepare_workload(output_path)

    def _prepare_workload(self, path):
        self.engine = self.workload_factory(self.torch, 'cuda_fp32_fma')
        self.quantum = self.engine.prepare(self.quantum_ms)
        anchor = self.engine.prepare(max(.5, self.quantum_ms * 10))
        delta_ms = anchor['observed_ms'] - self.quantum['observed_ms']
        if delta_ms <= 0:
            raise RuntimeError('invalid FMA timing slope')
        self.units_per_ms = (anchor['units'] - self.quantum['units']) / delta_ms
        if self.units_per_ms <= 0:
            raise RuntimeError('invalid FMA round slope')
        self.intercept_ms = max(0., self.quantum['observed_ms'] - self.quantum['units'] / self.units_per_ms)
        self.fma_model = dict(units_per_ms=self.units_per_ms, intercept_ms=self.intercept_ms, anchor=anchor)
        self.ready = True
        report = self.calibration_report()
        if path:
            Path(path).write_text(json.dumps(report, indent=2) + '\n')
        return report

    def _inject(self, key, op, units, phase):
        row = self.table.get(key)
        if row is None:
            row = self.fallbacks.get(key)
            if row is None:
                candidates = [v for v in self.table.values() if v['op'] == op and v['phase'] == phase]
                if not candidates:
                    raise RuntimeError(f'unseen compute operator after calibration: {op}')
                nearest = min(candidates, key=lambda v: abs(math.log(units / v['work_units'])))
                row = dict(op=op, phase=phase, work_units=units,
                           median_ms=nearest['median_ms'] * units / nearest['work_units'],
                           fallback='same_op_work_units_scaling', work_ratio=units / nearest['work_units'],
                           reference_signature=next(k for k, v in self.table.items() if v is nearest))
                self.fallbacks[key] = row
            self.stats['fallback_calls'] += 1
            self.iteration_fallbacks[key] += 1
        duration = row['median_ms']
        target = duration * self.extra_ratio
        self.stats['compute_calls'] += 1
        self.stats['healthy_estimated_ms'] += duration
        self.stats['requested_extra_ms'] += target
        detail = self.op_stats.setdefault(key, dict(signature=key, op=op, phase=phase,
            healthy_gpu_ms=duration, calls=0, requested_extra_ms=0., ratio=self.extra_ratio,
            fallback=key in self.fallbacks))
        detail['calls'] += 1
        detail['requested_extra_ms'] += target
        family = ('gemm' if op.split('.')[1] in ('mm', 'addmm', 'bmm', 'baddbmm', 'mv', 'dot', 'matmul', 'linear')
                  else 'attention' if 'attention' in op else 'other_local_compute')
        bucket = (phase, family)
        self.pending[bucket] += target
        bucket_detail = self.bucket_stats.setdefault(':'.join(bucket), dict(requested_extra_ms=0.,
            launch_rounds=0, injected_launches=0, calibrated_injected_ms=0.))
        bucket_detail['requested_extra_ms'] += target
        observed = self.quantum['observed_ms']
        if self.pending[bucket] >= observed:
            dose = self.pending[bucket]
            launch_units = max(1, round((dose - self.intercept_ms) * self.units_per_ms))
            if launch_units > 2_147_483_647:
                raise RuntimeError('FMA rounds exceed native int range')
            self.engine.launch_units(launch_units)
            predicted = self.intercept_ms + launch_units / self.units_per_ms
            self.pending[bucket] -= predicted
            bucket_detail['launch_rounds'] += launch_units
            bucket_detail['injected_launches'] += 1
            bucket_detail['calibrated_injected_ms'] += predicted
            self.stats['injected_launches'] += 1
            self.stats['calibrated_injected_ms'] += predicted

    def end_iteration(self):
        return dict(self.stats, catalog_sha256=getattr(self, 'catalog_sha256', None),
                    buckets=self.bucket_stats, pending_extra_ms=sum(self.pending.values()),
                    pending_buckets={':'.join(k): v for k, v in self.pending.items()},
                    operators=list(self.op_stats.values()),
                    fallback_shapes=[dict(signature=k, calls=v, **self.fallbacks[k])
                                     for k, v in self.iteration_fallbacks.items()])

    def calibration_report(self):
        return dict(rank=self.rank, catalog_sha256=self.catalog_sha256,
                    reference_catalog_sha256=getattr(self, 'reference_catalog_sha256', None),
                    reference_path=getattr(self, 'reference_path', None), kind='whole_local_compute_proportional',
                    samples_per_shape=self.samples_per_shape, quantum=self.quantum, fma_model=self.fma_model,
                    fallback_policy='same aten overload; nearest work estimate; scale by work ratio; fail if op unseen',
                    quantization='per-(autograd phase,compute family) cumulative credit; residual below one calibrated quantum per bucket plus round rounding',
                    compute_ops=sorted(COMPUTE_OPS), shapes=self.table,
                    excluded_cuda_ops=dict(self.ignored))
