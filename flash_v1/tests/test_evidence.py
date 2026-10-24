"""CPU fake-event protocol checks; GPU numerical/overhead validation is separate."""
import sys
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from flash_v1.evidence import EvidenceCollector, interval_residual


class Event:
    def __init__(self, cuda):
        self.cuda = cuda
        self.time = None
    def record(self, stream):
        self.time = self.cuda.time
    def query(self):
        return True
    def elapsed_time(self, other):
        return (other.time - self.time) / 1000


class FakeCuda:
    def __init__(self):
        self.time = 0.
        self.created = 0
    def Event(self, **kwargs):
        self.created += 1
        return Event(self)
    def current_stream(self):
        return SimpleNamespace(cuda_stream=1)


class Handle:
    def __init__(self, hooks, hook):
        self.hooks, self.hook = hooks, hook
    def remove(self):
        self.hooks.remove(self.hook)


class Module:
    def __init__(self):
        self.pre, self.post, self.bpre, self.bpost = [], [], [], []
    def _add(self, hooks, hook, prepend=False, **kwargs):
        hooks.insert(0, hook) if prepend else hooks.append(hook)
        return Handle(hooks, hook)
    def register_forward_pre_hook(self, hook, **kwargs):
        return self._add(self.pre, hook, **kwargs)
    def register_forward_hook(self, hook, **kwargs):
        return self._add(self.post, hook, **kwargs)
    def register_full_backward_pre_hook(self, hook, **kwargs):
        return self._add(self.bpre, hook, **kwargs)
    def register_full_backward_hook(self, hook, **kwargs):
        return self._add(self.bpost, hook, **kwargs)


class Qwen2MoeMLP(Module):
    pass


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cuda = FakeCuda()
        self.attention, self.expert = Module(), Qwen2MoeMLP()
        self.attention.layer_idx = 0
        mlp = SimpleNamespace(named_modules=lambda: [('experts.0', self.expert)])
        decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=self.attention, mlp=mlp)])
        self.collector = EvidenceCollector(SimpleNamespace(cuda=self.cuda), decoder, self.temp.name, 0)
        def allreduce(value, group, phase):
            self.cuda.time += 100
            return value
        self.tp = SimpleNamespace(_all_reduce_with_phase=allreduce)
        self.patch = patch.dict(sys.modules, {'flash_v1.tensor_parallel': self.tp})
        self.patch.start()
    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_injected_work_remains_but_internal_communication_is_removed(self):
        # Existing fault prehook is installed first. Evidence prepend must see it.
        self.attention.pre.append(lambda m, a, k: setattr(self.cuda, 'time', self.cuda.time + 5))
        original = self.tp._all_reduce_with_phase
        with self.collector.iteration(12, True):
            for hook in self.attention.pre:
                hook(self.attention, (), {})
            self.tp._all_reduce_with_phase(None, None, 1)
            self.cuda.time += 3
            for hook in self.attention.post:
                hook(self.attention, (), {}, None)
        self.assertIs(self.tp._all_reduce_with_phase, original)
        rows = self.collector.drain()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['module_elapsed_us'], 108)
        self.assertAlmostEqual(rows[0]['duration_us'], 8)
        self.assertTrue(rows[0]['coverage_complete'])
        self.assertEqual(len(self.attention.pre), 1)
        self.assertFalse(self.attention.bpre)
        created = self.cuda.created
        with self.collector.iteration(13, False):
            self.assertIs(self.tp._all_reduce_with_phase, original)
        self.assertEqual(self.collector.drain(), [])
        self.assertEqual(self.cuda.created, created)

    def test_missing_collective_and_scope_are_invalid(self):
        with self.collector.iteration(1, True):
            self.collector._begin('experts', 0, 'experts.0', 'backward', ())
            self.cuda.time += 5
            self.collector._end()
        self.assertFalse(self.collector.drain()[0]['coverage_complete'])
        with self.collector.iteration(2, True):
            self.collector._begin('attention', 0, 'attention', 'forward', ())
        self.assertFalse(self.collector.drain()[0]['coverage_complete'])

    def test_interval_union_and_scope_rejection(self):
        self.assertEqual(interval_residual(100, [(10, 40), (20, 60), (80, 90)]), 40)
        with self.assertRaises(ValueError):
            interval_residual(100, [(0, 101)])

    def test_attention_keyword_keeps_same_tensor_and_profile_marker(self):
        tensor = SimpleNamespace(shape=(1, 16, 32), dtype='float32')
        with self.collector.iteration(10, True, epoch=3):
            args, kwargs = self.attention.pre[0](self.attention, (),
                dict(hidden_states=tensor, position_ids=None))
            self.assertIs(args[0], tensor)
            self.assertEqual(kwargs, dict(position_ids=None))
            self.cuda.time += 1
            self.attention.post[0](self.attention, args, kwargs, None)
        self.collector.drain()
        lines = self.collector.path.read_text().splitlines()
        marker = json.loads(lines[-1])
        self.assertEqual(marker['record_type'], 'iteration_complete')
        self.assertEqual(marker['rows_count'], 1)
        self.assertEqual(marker['epoch'], 3)
        with self.collector.iteration(11, False):
            pass
        self.collector.drain()
        self.assertEqual(self.collector.path.read_text().splitlines(), lines)


if __name__ == '__main__':
    unittest.main()
