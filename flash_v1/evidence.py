"""Opt-in, sparse PROFILE-only module residuals; no TorchDispatch or added sync.

Wrap training with ``collector.iteration(step, enabled, epoch=epoch)`` and call
``drain()`` after the existing CUDA completion boundary. No hooks, monkeypatch,
events or output occur in disabled iterations. Residuals include stream idle and
local CPU enqueue gaps; they are not physical kernel-only compute durations.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time


def interval_residual(total, intervals):
    """Union subtraction, rejecting incomplete/out-of-scope intervals."""
    if total <= 0:
        raise ValueError('nonpositive module duration')
    tail = used = 0.
    for left, right in sorted(intervals):
        if left < -.01 or right > total + .01 or right < left:
            raise ValueError('communication interval outside module')
        left, right = max(0., left), min(total, right)
        used += max(0., right - max(left, tail))
        tail = max(tail, right)
    return max(0., total - used)


class EvidenceCollector:
    def __init__(self, torch, decoder, output, rank, selection_index=0):
        self.torch, self.decoder, self.rank = torch, decoder, rank
        self.path = Path(output) / f'evidence_rank{rank}.jsonl'
        self.selection_index = selection_index
        self.pending, self.pool, self.handles = [], [], []
        self.local = threading.local()
        self.active = False
        self.drain_meta = None
        self.modules = {'attention': [], 'experts': []}
        for layer in decoder.layers:
            index = layer.self_attn.layer_idx
            self.modules['attention'].append((index, 'attention', layer.self_attn))
            for name, module in layer.mlp.named_modules():
                # Leaf expert modules exclude routing and EP dispatch/return.
                if module.__class__.__name__ == 'Qwen2MoeMLP':
                    self.modules['experts'].append((index, name, module))
        if not all(self.modules.values()):
            raise ValueError('evidence requires attention and owned expert modules')

    def _stack(self):
        if not hasattr(self.local, 'stack'):
            self.local.stack = []
        return self.local.stack

    def _event(self):
        event = self.pool.pop() if self.pool else self.torch.cuda.Event(enable_timing=True)
        event.record(self.torch.cuda.current_stream())
        return event

    def _stream(self):
        return int(self.torch.cuda.current_stream().cuda_stream)

    def _begin(self, kind, layer, name, phase, tensors):
        key = (kind, phase)
        occurrence = self.counts.get(key, 0)
        self.counts[key] = occurrence + 1
        # One microbatch per selected module and direction; fixed in a window.
        if occurrence:
            self._stack().append(None)
            return
        tensor = next((x for x in tensors if hasattr(x, 'shape')), None)
        row = dict(schema_version=1, record_type='measurement', rank=self.rank, iteration=self.step,
                   epoch=self.epoch, layer=layer, phase=phase, slot=kind,
                   module_kind=kind, module_name=name, occurrence=occurrence,
                   shape=list(tensor.shape) if tensor is not None else [],
                   dtype=str(tensor.dtype) if tensor is not None else None,
                   measurement='local_residual', stream_id=self._stream(),
                   coverage_complete=True, comms=[], begin=self._event(),
                   end=None, cpu_begin=time.perf_counter_ns())
        self.pending.append(row)
        self._stack().append(row)

    def _end(self):
        row = self._stack().pop()
        if row is None:
            return
        row['end'] = self._event()
        row['cpu_enqueue_us'] = (time.perf_counter_ns() - row.pop('cpu_begin')) / 1000
        row['coverage_complete'] &= self._stream() == row['stream_id']

    def _attach(self, kind, layer, name, module):
        def fwd_pre(_module, args, kwargs):
            # HF calls attention with hidden_states as a keyword. PyTorch's
            # full backward hook tracks positional inputs only; with kwargs it
            # fires its post hook immediately at output-gradient arrival.
            # Pass the same tensor to the same first parameter position, so the
            # hook spans the actual Q/K/V input gradient computation. This is
            # PROFILE-only and does not place a hook on the residual branch.
            if kind == 'attention' and not args and 'hidden_states' in kwargs:
                kwargs = dict(kwargs)
                args = (kwargs.pop('hidden_states'),)
            self._begin(kind, layer, name, 'forward', (*args, *kwargs.values()))
            return args, kwargs
        def fwd_post(_module, args, kwargs, output):
            self._end()
        def bwd_pre(_module, gradients):
            self._begin(kind, layer, name, 'backward', gradients)
        def bwd_post(_module, grad_input, grad_output):
            self._end()
        self.handles.extend([
            module.register_forward_pre_hook(fwd_pre, prepend=True, with_kwargs=True),
            module.register_forward_hook(fwd_post, with_kwargs=True, always_call=True),
            module.register_full_backward_pre_hook(bwd_pre, prepend=True),
            module.register_full_backward_hook(bwd_post),
        ])

    @contextmanager
    def iteration(self, step, enabled, epoch=1):
        if not enabled:
            yield self
            return
        if self.active or self.pending:
            raise RuntimeError('nested or undrained evidence iteration')
        from flash_v1 import tensor_parallel
        self.step, self.epoch, self.counts, self.active = step, epoch, {}, True
        self.drain_meta = dict(record_type='iteration_complete', rank=self.rank,
                               iteration=step, epoch=epoch)
        original = tensor_parallel._all_reduce_with_phase
        def measured(value, group, phase):
            stack = self._stack()
            row = stack[-1] if stack else None
            if row is None:
                return original(value, group, phase)
            start, stream = self._event(), self._stream()
            cpu_start = time.perf_counter_ns()
            try:
                return original(value, group, phase)
            finally:
                end = self._event()
                row['coverage_complete'] &= stream == row['stream_id'] == self._stream()
                row['comms'].append((start, end, phase,
                                     (time.perf_counter_ns() - cpu_start) / 1000))
        try:
            # Same semantic selection across ranks and within all 8 profile steps.
            for kind, candidates in self.modules.items():
                layer, name, module = candidates[self.selection_index % len(candidates)]
                self._attach(kind, layer, name, module)
            tensor_parallel._all_reduce_with_phase = measured
            yield self
        finally:
            tensor_parallel._all_reduce_with_phase = original
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self.active = False
            if self._stack():
                for row in self.pending:
                    row['coverage_complete'] = False
                self._stack().clear()

    def drain(self):
        """Read completed events only; never synchronize or silently drop pending."""
        if self.active:
            raise RuntimeError('drain after training context and existing CUDA sync')
        rows = []
        pending, self.pending = self.pending, []
        for row in pending:
            begin, end, comms = row.pop('begin'), row.pop('end'), row.pop('comms')
            events = [begin] + ([end] if end is not None else [])
            events += [event for left, right, _, _ in comms for event in (left, right)]
            complete = end is not None and all(event.query() for event in events)
            row['coverage_complete'] &= complete
            intervals = []
            if complete:
                row['module_elapsed_us'] = begin.elapsed_time(end) * 1000
                for left, right, phase, cpu_us in comms:
                    intervals.append(dict(start_us=begin.elapsed_time(left) * 1000,
                                          end_us=begin.elapsed_time(right) * 1000,
                                          phase_code=phase, cpu_enqueue_us=cpu_us))
                try:
                    row['duration_us'] = interval_residual(row['module_elapsed_us'],
                        [(x['start_us'], x['end_us']) for x in intervals])
                except ValueError:
                    row['coverage_complete'] = False
                    row['duration_us'] = None
                expected = (1 if row['phase'] == 'forward' else
                            3 if row['module_kind'] == 'attention' else 2)
                row['coverage_complete'] &= len(comms) == expected
            else:
                row['duration_us'] = None
                row['invalid_reason'] = 'incomplete_scope_or_events_not_ready'
            row['communication_intervals'] = intervals
            row['event_pairs'] = 1 + len(comms)
            rows.append(row)
            # Reuse only events known complete. Missing end is an invalid sample.
            if complete:
                self.pool.extend(events)
        marker, self.drain_meta = self.drain_meta, None
        if marker is not None:
            marker.update(rows_count=len(rows), expected_rows=4,
                          event_pairs=sum(row['event_pairs'] for row in rows))
        if rows or marker is not None:
            with self.path.open('a') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
                if marker is not None:
                    stream.write(json.dumps(marker) + '\n')
        return rows
