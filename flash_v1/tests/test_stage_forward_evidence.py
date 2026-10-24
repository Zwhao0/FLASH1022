"""CPU protocol tests; no torch import, GPU execution, or training changes."""
import math
from types import SimpleNamespace
import unittest

from flash_v1.stage_forward_evidence import StageForwardEvidence, strict_residual


class Event:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self, stream):
        self.time = self.cuda.time

    def query(self):
        return self.cuda.ready

    def elapsed_time(self, other):
        return (other.time - self.time) / 1000


class Cuda:
    def __init__(self):
        self.time, self.created, self.stream, self.ready = 0., 0, 1, True

    def Event(self, **kwargs):
        self.created += 1
        return Event(self)

    def current_stream(self):
        return SimpleNamespace(cuda_stream=self.stream)


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook, **kwargs):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def observe(self, shape):
        tensor = SimpleNamespace(shape=shape, dtype='bfloat16')
        for hook in self.hooks:
            assert hook(self, (tensor,), {}) is None
        return tensor


class Qwen2MoeMLP(Module):
    pass


class StageForwardTests(unittest.TestCase):
    def setUp(self):
        self.cuda, self.stage, self.mlp = Cuda(), Module(), Qwen2MoeMLP()
        layer = SimpleNamespace(self_attn=SimpleNamespace(layer_idx=12),
            mlp=SimpleNamespace(named_modules=lambda: [('experts.4', self.mlp)]))
        self.stage.decoder = SimpleNamespace(layers=[layer])
        def exchange(*args, **kwargs):
            self.cuda.time += 3
            return 'preserved collective result'
        self.dist = SimpleNamespace(**{name: exchange for name in
            ('all_reduce', 'all_to_all_single', 'all_gather')})
        self.include_metadata, self.extra, self.calls = True, 0., 0
        def forward(*args, **kwargs):
            self.calls += 1
            self.stage.observe((1, 16))
            self.mlp.observe((self.calls, 128))
            self.cuda.time += 2 + (self.extra if self.calls == 2 else 0)
            if self.include_metadata:
                with self.collector.ep_metadata():
                    self.dist.all_to_all_single(None, None)
                    self.cuda.time += 4  # Existing two .tolist() host wait.
            else:
                self.dist.all_to_all_single(None, None)
                self.cuda.time += 4
            self.dist.all_reduce(None)
            self.dist.all_gather(None, None)
            self.cuda.time += 11
            return args, kwargs
        self.original_forward = forward
        self.schedules = SimpleNamespace(forward_step=forward)
        self.collector = StageForwardEvidence(SimpleNamespace(cuda=self.cuda), self.stage,
            self.schedules, self.dist, 7, expected_collectives=dict(
                all_reduce=1, all_to_all_single=1, all_gather=1))

    def run_two(self):
        for _ in range(2):
            self.assertEqual(self.schedules.forward_step('same tensor', flag=1),
                             (('same tensor',), {'flag': 1}))
            self.cuda.time += 1000  # PP/BWD outside FWD must never enter sums.
            self.dist.all_reduce(None)  # Unobserved outside FWD.

    def test_complete_two_mb_union_work_and_second_mb_fault(self):
        self.extra = 17
        with self.collector.iteration(3, enabled=True, epoch=2):
            self.run_two()
            with self.assertRaises(RuntimeError):
                self.collector.drain()
        result = self.collector.drain()
        self.assertTrue(result['coverage_complete'], result['invalid_reasons'])
        self.assertEqual(result['duration_us'], 43)
        self.assertEqual(result['event_pairs'], 10)
        self.assertEqual(result['microbatches'][1]['work']['12:experts.4'][0]['shape'], [2, 128])
        self.assertEqual(result['epoch'], 2)
        self.assertIs(self.schedules.forward_step, self.original_forward)
        self.assertFalse(self.stage.hooks or self.mlp.hooks)
        created = self.cuda.created
        self.calls = 0
        with self.collector.iteration(4, enabled=True):
            self.run_two()
        self.assertTrue(self.collector.drain()['coverage_complete'])
        self.assertEqual(created, self.cuda.created)  # Completed event pool reused.

    def test_disabled_default_is_inert(self):
        originals = dict(vars(self.dist))
        with self.collector.iteration(1):
            self.assertIs(self.schedules.forward_step, self.original_forward)
            self.assertEqual(vars(self.dist), originals)
            self.assertFalse(self.stage.hooks or self.mlp.hooks)
            with self.collector.ep_metadata():
                pass
        self.assertEqual(self.cuda.created, 0)
        self.assertIsNone(self.collector.drain())

    def test_missing_metadata_and_microbatch_fail_closed(self):
        self.include_metadata = False
        with self.collector.iteration(1, enabled=True):
            self.schedules.forward_step()
        result = self.collector.drain()
        self.assertFalse(result['coverage_complete'])
        self.assertIsNone(result['duration_us'])
        self.assertIn('incomplete_ep_metadata_coverage', result['invalid_reasons'])
        self.assertIn('expected_exactly_two_forward_microbatches', result['invalid_reasons'])

    def test_async_and_stream_change_fail_closed(self):
        original = self.original_forward
        def altered(*args, **kwargs):
            result = original(*args, **kwargs)
            self.dist.all_reduce(None, async_op=True)
            self.cuda.stream = 2
            return result
        self.schedules.forward_step = altered
        with self.collector.iteration(1, enabled=True):
            self.run_two()
        result = self.collector.drain()
        self.assertIsNone(result['duration_us'])
        for reason in ('asynchronous_collective_not_supported', 'stream_changed', 'collective_manifest_mismatch'):
            self.assertIn(reason, result['invalid_reasons'])

    def test_not_ready_does_not_sync_or_reuse_events(self):
        with self.collector.iteration(1, enabled=True):
            self.run_two()
        self.cuda.ready = False
        result = self.collector.drain()
        self.assertFalse(result['coverage_complete'])
        self.assertIn('events_not_ready_or_missing', result['invalid_reasons'])
        self.assertFalse(self.collector.pool)

    def test_exception_restores_wrappers_and_preserves_error(self):
        original = self.dist.all_reduce
        with self.assertRaisesRegex(ValueError, 'training failed'):
            with self.collector.iteration(1, enabled=True):
                self.schedules.forward_step()
                raise ValueError('training failed')
        self.assertIs(self.dist.all_reduce, original)
        self.assertIs(self.schedules.forward_step, self.original_forward)
        self.assertFalse(self.stage.hooks or self.mlp.hooks)
        self.assertFalse(self.collector.drain()['coverage_complete'])

    def test_union_rejects_outside_negative_nonfinite_and_accepts_nested(self):
        self.assertEqual(strict_residual(100, [(10, 40), (20, 60), (80, 90)]), 40)
        self.assertEqual(strict_residual(100, [(0, 100), (20, 30)]), 0)
        for total, pairs in ((-1, []), (math.nan, []), (100, [(-.001, 1)]),
                             (100, [(2, 1)]), (100, [(0, 100.001)]),
                             (100, [(0, math.inf)])):
            with self.assertRaises(ValueError):
                strict_residual(total, pairs)


if __name__ == '__main__':
    unittest.main()
