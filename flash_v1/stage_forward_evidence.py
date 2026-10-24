"""Unused, opt-in full-microbatch FWD evidence; no autograd or CUDA sync.

Call iteration(..., enabled=True) around the native schedule and drain() after
the caller's existing CUDA completion boundary. Requires an explicit collective
manifest and an ep_metadata() integration around each EP count exchange AND its
two .tolist() calls. Without that integration samples fail closed. Two separate
forward_step spans are summed, never the intervening pipeline/backward time.
This is a stream residual, including CPU enqueue gaps, not kernel-only time.
"""
from contextlib import contextmanager
import math
import threading


COLLECTIVES = ('all_reduce', 'all_to_all_single', 'all_gather')
SCHEMA = 'flash_stage_forward_aux_v1'


def strict_residual(total, intervals):
    """Subtract a union; reject nonfinite, reversed, outside or negative data."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not finite(total) or total <= 0:
        raise ValueError('invalid forward duration')
    pairs = list(intervals)
    for left, right in pairs:
        if not finite(left) or not finite(right) or not 0 <= left <= right <= total:
            raise ValueError('invalid communication interval')
    tail, lengths = 0., []
    for left, right in sorted(pairs):
        lengths.append(max(0., right - max(left, tail)))
        tail = max(tail, right)
    residual = total - math.fsum(lengths)
    if residual < 0:
        raise ValueError('negative forward residual')
    return residual


class StageForwardEvidence:
    """Dependency-injected collector; construction and disabled iterations inert.

    expected_collectives is the exact per-microbatch count for each of the
    three intercepted dist APIs, including zero. metadata_spans defaults to one
    per decoder layer. stage_metadata is caller-owned topology/model metadata,
    never a fault plan. This object owns process-local temporary wrappers; do
    not overlap it with another wrapper of the same APIs.
    """
    def __init__(self, torch, stage, schedules, dist, rank, *,
                 expected_collectives, stage_metadata=None, metadata_spans=None):
        if set(expected_collectives) != set(COLLECTIVES) or any(
                type(value) is not int or value < 0 for value in expected_collectives.values()):
            raise ValueError('exact nonnegative collective manifest required')
        self.torch, self.stage, self.schedules, self.dist = torch, stage, schedules, dist
        self.rank, self.expected = rank, dict(expected_collectives)
        self.metadata = dict(stage_metadata or {})
        self.modules = {}
        for layer in stage.decoder.layers:
            index = layer.self_attn.layer_idx
            for name, module in layer.mlp.named_modules():
                if type(module).__name__ == 'Qwen2MoeMLP':
                    self.modules[f'{index}:{name}'] = module
        if not self.modules:
            raise ValueError('no owned/shared Qwen2MoeMLP work population')
        self.metadata_spans = len(stage.decoder.layers) if metadata_spans is None else metadata_spans
        if type(self.metadata_spans) is not int or self.metadata_spans < 0:
            raise ValueError('invalid EP metadata span count')
        self.active, self.pending, self.current = False, None, None
        self.pool = []

    def _invalid(self, reason):
        if reason not in self.pending['invalid_reasons']:
            self.pending['invalid_reasons'].append(reason)

    def _event(self):
        event = self.pool.pop() if self.pool else self.torch.cuda.Event(enable_timing=True)
        stream = self.torch.cuda.current_stream()
        event.record(stream)
        if self.current is not None and int(stream.cuda_stream) != self.current['stream_id']:
            self._invalid('stream_changed')
        return event

    def _observing(self):
        if self.current is None:
            return False
        if threading.get_ident() != self.owner:
            self._invalid('cross_thread_forward_work')
            return False
        return True

    def _shape(self, tensor):
        shape = getattr(tensor, 'shape', None)
        if shape is None:
            self._invalid('missing_tensor_shape')
            return None
        return dict(shape=list(shape), dtype=str(tensor.dtype))

    def _work_hook(self, key):
        def observe(_module, args, kwargs):
            if self._observing():
                tensor = args[0] if args else kwargs.get('hidden_states', kwargs.get('x'))
                self.current['work'].setdefault(key, []).append(self._shape(tensor))
            # None preserves arguments and Tensor identity.
        return observe

    def _stage_hook(self, _module, args, kwargs):
        if self._observing():
            tensor = args[0] if args else kwargs.get('input_ids')
            self.current['stage_inputs'].append(self._shape(tensor))

    def _collective_wrapper(self, name, original):
        def measured(*args, **kwargs):
            if not self._observing():
                return original(*args, **kwargs)
            index = 5 if name == 'all_to_all_single' else 3
            if kwargs.get('async_op', args[index] if len(args) > index else False):
                self._invalid('asynchronous_collective_not_supported')
            row = self.current
            interval = dict(kind=name, begin=self._event(), end=None)
            row['intervals'].append(interval)
            try:
                return original(*args, **kwargs)
            except BaseException:
                self._invalid('collective_exception')
                raise
            finally:
                interval['end'] = self._event()
        return measured

    @contextmanager
    def ep_metadata(self):
        """Wrap exactly count exchange plus BOTH .tolist(), excluding compute."""
        if not self.active or not self._observing():
            yield
            return
        row = self.current
        if row.get('metadata_active'):
            self._invalid('nested_ep_metadata_span')
            yield
            return
        row['metadata_active'] = True
        interval = dict(kind='ep_metadata', begin=self._event(), end=None)
        row['intervals'].append(interval)
        before = sum(item['kind'] == 'all_to_all_single' for item in row['intervals'])
        try:
            yield
        except BaseException:
            self._invalid('ep_metadata_exception')
            raise
        finally:
            interval['end'] = self._event()
            row['metadata_active'] = False
            after = sum(item['kind'] == 'all_to_all_single' for item in row['intervals'])
            if after - before != 1:
                self._invalid('metadata_span_requires_one_count_exchange')

    def _forward_wrapper(self, original):
        def measured(*args, **kwargs):
            if threading.get_ident() != self.owner or self.current is not None:
                self._invalid('nested_or_cross_thread_forward')
                return original(*args, **kwargs)
            row = dict(microbatch=len(self.pending['microbatches']),
                       stream_id=int(self.torch.cuda.current_stream().cuda_stream),
                       intervals=[], work={}, stage_inputs=[], begin=None, end=None)
            self.pending['microbatches'].append(row)
            self.current = row
            row['begin'] = self._event()
            try:
                return original(*args, **kwargs)
            except BaseException:
                self._invalid('forward_exception')
                raise
            finally:
                row['end'] = self._event()
                self.current = None
        return measured

    @contextmanager
    def iteration(self, step, enabled=False, epoch=1):
        if not enabled:
            yield self
            return
        if self.active or self.pending is not None:
            raise RuntimeError('nested or undrained stage forward iteration')
        self.owner, self.active = threading.get_ident(), True
        self.pending = dict(schema=SCHEMA, record_type='stage_forward_aux',
            rank=self.rank, iteration=step, epoch=epoch, expected_microbatches=2,
            expected_collectives=dict(self.expected), expected_ep_metadata_spans=self.metadata_spans,
            stage_metadata=dict(self.metadata), microbatches=[], invalid_reasons=[])
        handles, patches = [], []
        try:
            handles.append(self.stage.register_forward_pre_hook(self._stage_hook, with_kwargs=True))
            for key, module in self.modules.items():
                handles.append(module.register_forward_pre_hook(self._work_hook(key), with_kwargs=True))
            for name in COLLECTIVES:
                original = getattr(self.dist, name)
                wrapper = self._collective_wrapper(name, original)
                patches.append((self.dist, name, original, wrapper))
                setattr(self.dist, name, wrapper)
            original = self.schedules.forward_step
            wrapper = self._forward_wrapper(original)
            patches.append((self.schedules, 'forward_step', original, wrapper))
            self.schedules.forward_step = wrapper
            yield self
        except BaseException:
            self._invalid('iteration_exception')
            raise
        finally:
            for owner, name, original, wrapper in reversed(patches):
                if getattr(owner, name) is wrapper:
                    setattr(owner, name, original)
                else:
                    self._invalid('temporary_wrapper_replaced')
            for handle in handles:
                handle.remove()
            self.active = False

    def drain(self):
        """Query only, after existing CUDA sync; incomplete samples are invalid."""
        if self.active:
            raise RuntimeError('drain after iteration and existing CUDA completion boundary')
        if self.pending is None:
            return None
        result = self.pending
        if len(result['microbatches']) != 2:
            self._invalid('expected_exactly_two_forward_microbatches')
        durations, reusable = [], []
        for row in result['microbatches']:
            begin, end = row.pop('begin'), row.pop('end')
            raw = row.pop('intervals')
            row.pop('metadata_active', None)
            events = [begin, end] + [event for item in raw for event in (item['begin'], item['end'])]
            ready = all(event is not None and event.query() for event in events)
            row.update(duration_us=None, forward_elapsed_us=None, communication_intervals=[],
                       event_pairs=1 + len(raw))
            counts = {name: sum(item['kind'] == name for item in raw) for name in COLLECTIVES}
            row['collective_counts'] = counts
            if counts != self.expected:
                self._invalid('collective_manifest_mismatch')
            if sum(item['kind'] == 'ep_metadata' for item in raw) != self.metadata_spans:
                self._invalid('incomplete_ep_metadata_coverage')
            if len(row['stage_inputs']) != 1 or row['stage_inputs'][0] is None:
                self._invalid('expected_one_stage_model_call_per_microbatch')
            if set(row['work']) != set(self.modules) or any(
                    len(values) != 1 or values[0] is None for values in row['work'].values()):
                self._invalid('incomplete_or_duplicate_mlp_work_vector')
            if not ready:
                self._invalid('events_not_ready_or_missing')
                continue
            reusable.extend(events)
            total = begin.elapsed_time(end) * 1000
            row['forward_elapsed_us'] = total
            for item in raw:
                row['communication_intervals'].append(dict(kind=item['kind'],
                    start_us=begin.elapsed_time(item['begin']) * 1000,
                    end_us=begin.elapsed_time(item['end']) * 1000))
            try:
                residual = strict_residual(total, [(item['start_us'], item['end_us'])
                    for item in row['communication_intervals']])
            except ValueError:
                self._invalid('invalid_communication_union_or_residual')
            else:
                row['duration_us'] = residual
                durations.append(residual)
        result['coverage_complete'] = not result['invalid_reasons']
        result['duration_us'] = math.fsum(durations) if result['coverage_complete'] else None
        result['event_pairs'] = sum(row['event_pairs'] for row in result['microbatches'])
        self.pool.extend(reusable)
        self.pending = None
        return result
