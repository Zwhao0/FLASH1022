"""CPU fake dispatch: timing, phase, uniform ratio, fallback and credit conservation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from flash_v1.whole_compute import WholeComputeSlowdown


class Tensor:
    dtype = 'float32'
    device = SimpleNamespace(type='cuda')
    def __init__(self, *shape):
        self.shape = shape
    def stride(self):
        return (self.shape[-1], 1)


class Operator:
    def __init__(self, state, name='aten.mm.default'):
        self.state, self.name = state, name
    def __str__(self):
        return self.name
    def __call__(self, *args, **kwargs):
        self.state.clock += .1
        return args[0]


class Workload:
    def __init__(self, torch, engine):
        self.launches = []
    def prepare(self, ms):
        return dict(units=round(ms * 1000), observed_ms=ms, requested_ms=ms)
    def launch_units(self, units):
        self.launches.append(units)


def setup():
    state = SimpleNamespace(clock=0., events=0, syncs=0, phase=-1)
    class Event:
        def __init__(self, **kwargs):
            state.events += 1
        def record(self):
            self.value = state.clock
        def elapsed_time(self, end):
            return end.value - self.value
    def sync():
        state.syncs += 1
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=sync),
        _C=SimpleNamespace(_current_graph_task_id=lambda: state.phase))
    mode = WholeComputeSlowdown(torch, workload_factory=Workload)
    return state, mode


class WholeComputeTests(unittest.TestCase):
    def test_calibrate_median_frozen_no_runtime_events_uniform_ratio_and_fallback(self):
        state, mode = setup()
        op = Operator(state)
        args = (Tensor(4, 8), Tensor(8, 16))
        mode.begin_calibration()
        for _ in range(5):
            mode.dispatch(op, args)
        state.phase = 7
        mode.dispatch(op, args)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            report = mode.finish_calibration(path)
            self.assertTrue(path.exists())
            _, reused = setup()
            reused.load_calibration(path, Path(directory) / 'copied.json')
            self.assertEqual(reused.catalog_sha256, report['catalog_sha256'])
            self.assertEqual(reused.reference_catalog_sha256, report['catalog_sha256'])
        self.assertEqual(sorted(len(v['samples_ms']) for v in report['shapes'].values()), [1, 3])
        before = state.events, state.syncs
        mode.begin_iteration(10, extra_ratio=2.)
        result = mode.dispatch(op, args)
        self.assertIs(result, args[0])
        # Previously unseen routing token count scales GEMM work by exactly 2.
        mode.dispatch(op, (Tensor(8, 8), Tensor(8, 16)))
        row = mode.end_iteration()
        self.assertEqual((state.events, state.syncs), before)
        self.assertAlmostEqual(row['requested_extra_ms'], .6)
        self.assertAlmostEqual(row['requested_extra_ms'], row['calibrated_injected_ms'] +
                               row['pending_extra_ms'] - row['pending_before_ms'])
        self.assertEqual(row['fallback_calls'], 1)
        self.assertEqual(row['fallback_shapes'][0]['work_ratio'], 2.)
        self.assertTrue(all(v['phase'] == 'bwd' for v in row['operators']))
        for operator in row['operators']:
            self.assertAlmostEqual(operator['requested_extra_ms'], operator['healthy_gpu_ms'] * operator['calls'] * 2)

    def test_credit_is_shared_across_shapes_and_communication_excluded(self):
        state, mode = setup()
        op = Operator(state)
        mode.begin_calibration()
        mode.dispatch(op, (Tensor(100, 8), Tensor(8, 16)))
        mode.dispatch(Operator(state, 'c10d.allreduce_.default'), (Tensor(4, 4),))
        mode.finish_calibration()
        mode.begin_iteration(20, .5)
        for i in range(1, 21):
            mode.dispatch(op, (Tensor(i, 8), Tensor(8, 16)))
        row = mode.end_iteration()
        self.assertEqual(len(row['pending_buckets']), 1)
        self.assertLess(abs(row['pending_extra_ms']), .05)
        self.assertGreater(row['injected_launches'], 0)
        self.assertIn('c10d.allreduce_.default', mode.ignored)
        self.assertAlmostEqual(row['requested_extra_ms'], row['calibrated_injected_ms'] + row['pending_extra_ms'])
        with self.assertRaisesRegex(RuntimeError, 'unseen compute operator'):
            mode.dispatch(Operator(state, 'aten.bmm.default'), (Tensor(1, 2, 3), Tensor(1, 3, 4)))

    def test_requires_calibration_and_valid_ratio(self):
        _, mode = setup()
        with self.assertRaises(RuntimeError):
            mode.begin_iteration(1, 1)
        for value in [-1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                mode.begin_iteration(1, value)


if __name__ == '__main__':
    unittest.main()
