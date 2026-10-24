"""CPU-only hook/dose checks, without importing CUDA or training dependencies."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from flash_v1.fault_workloads import MODES, StreamWorkload


class Module:
    def __init__(self):
        self.forward, self.backward = [], []

    def register_forward_pre_hook(self, hook):
        self.forward.append(hook)
        return SimpleNamespace(remove=lambda: self.forward.remove(hook))

    def register_full_backward_pre_hook(self, hook):
        self.backward.append(hook)
        return SimpleNamespace(remove=lambda: self.backward.remove(hook))

    def fire(self, phase):
        for hook in getattr(self, phase):
            self.assert_return = hook(self, ())
            assert self.assert_return is None


class Workload:
    def __init__(self, torch, engine):
        self.engine, self.prepared, self.launches = engine, {}, []

    def prepare(self, dose):
        self.prepared[dose] = dict(units=round(dose * 1000), requested_ms=dose,
                                   observed_ms=dose, engine=self.engine, cycles_per_ms=1000)
        return self.prepared[dose]

    def launch(self, dose):
        self.launches.append(dose)
        return self.prepared[dose]


def load_collection():
    torch = ModuleType('torch')
    torch.nn = SimpleNamespace(Identity=type('Identity', (), {}))
    dist = ModuleType('torch.distributed')
    dist.get_rank = lambda: 0
    torch.distributed = dist
    tp = ModuleType('flash_v1.tensor_parallel')
    tp.set_collection_context = lambda context: None
    spec = importlib.util.spec_from_file_location('fault_test_collection',
        Path(__file__).parents[1] / 'collection.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist,
                                 'flash_v1.tensor_parallel': tp}):
        spec.loader.exec_module(module)
    module.StreamWorkload = Workload
    return module


class FaultWorkloadTests(unittest.TestCase):
    def test_all_modes_repeated_strengths_legacy_and_truth_isolation(self):
        collection = load_collection()
        rows = [dict(start=1, end=1, target_ranks=[], delay_ms=0)]
        modes = [m for m in MODES if m != 'whole_compute'] + [MODES[0], None]
        for i, mode in enumerate(modes, 2):
            row = dict(start=i, end=i, target_ranks=[0], delay_ms=i * 10)
            if mode:
                row['fault_mode'] = mode
            rows.append(row)
        rows.append(dict(start=8, end=8, target_ranks=[1], delay_ms=80,
                         fault_mode='attention_bwd_compute'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '0.bin').write_bytes(bytes(16))
            (root / 'plan.json').write_text(json.dumps(dict(intervals=rows,
                profile_enabled=True, audit_telemetry=True)))
            args = SimpleNamespace(collection_plan=str(root / 'plan.json'),
                                   delay_placement='attention-experts')
            with patch.dict(os.environ, FLASH_CONTEXT_DIR=directory):
                session = collection.CollectionSession(args, root)
            attention, experts = [Module() for _ in range(2)], [Module() for _ in range(4)]
            layers = [SimpleNamespace(self_attn=attention[i],
                       mlp=SimpleNamespace(experts=experts[i * 2:i * 2 + 2])) for i in range(2)]
            session.attach_backward(SimpleNamespace(model=SimpleNamespace(layers=layers)))
            engines = session.engines
            # No GPU work was launched in healthy/unselected-rank iterations.
            for row in rows:
                i = row['start']
                session.begin_iteration(i)
                for _ in range(2):
                    for module in attention + experts:
                        module.fire('forward')
                        module.fire('backward')
                targeted = row['target_ranks'] == [0]
                self.assertEqual(session.estimated_ms, 2 * row['delay_ms'] if targeted else 0)
                if targeted:
                    mode = row.get('fault_mode')
                    calls = 8 if mode == 'experts_bwd_sleep' else 12 if mode is None else 4
                    self.assertEqual(session.calls, calls)
                session.end_iteration(i, 123)
            session.close(8)
            truth = [json.loads(line) for line in (root / 'injection_rank0.jsonl').read_text().splitlines()]
            events = [row for row in truth if 'hook_calls' in row]
            self.assertEqual(len(events), 6)
            compute = next(row for row in events if row['engine'] == 'cuda_fp32_fma')
            self.assertEqual(compute['injected_cycles'], 0)
            self.assertEqual(compute['calibrated_injected_ms'], 100)
            telemetry = [json.loads(line) for line in (root / 'telemetry_rank0.jsonl').read_text().splitlines()]
            self.assertEqual(len(telemetry), 8)
            self.assertEqual(set(telemetry[0]), {'rank', 'iteration', 'time_ms', 'monotonic_us'})
            self.assertTrue(all(not module.forward and not module.backward for module in attention + experts))
            self.assertEqual(len(engines['cuda_fp32_fma'].launches), 4)

    def test_sleep_calibration_is_cached_and_launch_has_no_events(self):
        state = SimpleNamespace(units=0, events=0)
        class Event:
            def __init__(self, **kwargs):
                state.events += 1
            def record(self):
                pass
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return state.units / 1000
        def sleep(units):
            state.units = units
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, _sleep=sleep))
        engine = StreamWorkload(torch, 'cuda_sleep')
        engine.prepare(4)
        before = state.events
        self.assertEqual(engine.prepare(4)['observed_ms'], 4)
        engine.launch(4)
        self.assertEqual(state.events, before)
        self.assertEqual(state.units, 4000)

    def test_compute_uses_stream_buffer_and_rounds_without_runtime_allocation(self):
        state = SimpleNamespace(units=0, events=0, allocations=0, calls=[])
        class Event:
            def __init__(self, **kwargs):
                state.events += 1
            def record(self):
                pass
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return state.units / 1000
        def launch(stream, output, blocks, rounds):
            state.units = rounds
            state.calls.append((stream, output, blocks, rounds))
            return 0
        def empty(size, **kwargs):
            state.allocations += 1
            self.assertEqual(size, 20 * 128)
            return SimpleNamespace(data_ptr=lambda: 456)
        torch = SimpleNamespace(float32='float32', empty=empty, cuda=SimpleNamespace(
            Event=Event, current_device=lambda: 0,
            get_device_properties=lambda device: SimpleNamespace(multi_processor_count=10),
            current_stream=lambda: SimpleNamespace(cuda_stream=123)))
        with patch('flash_v1.fault_workloads.ctypes.CDLL', return_value=SimpleNamespace(
                flash_compute_launch=SimpleNamespace())) as loader:
            # A callable with ctypes-style signature attributes.
            loader.return_value.flash_compute_launch = launch
            engine = StreamWorkload(torch, 'cuda_fp32_fma')
            result = engine.prepare(5)
            self.assertEqual(result['rounds'], 5000)
            before = (state.events, state.allocations)
            engine.launch(5)
            self.assertEqual((state.events, state.allocations), before)
            self.assertEqual(state.calls[-1], (123, 456, 20, 5000))

    def test_non_target_rank_does_not_construct_engine_or_attach_hooks(self):
        collection = load_collection()
        session = collection.CollectionSession.__new__(collection.CollectionSession)
        session.rank, session.hooks, session.engines = 0, [], {}
        session.intervals = [dict(target_ranks=[1], delay_ms=100, fault_mode='attention_bwd_compute')]
        session.args = SimpleNamespace(delay_placement='attention-experts')
        session.truth = io.StringIO()
        attention = Module()
        session.attach_backward(SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(self_attn=attention, mlp=SimpleNamespace(experts=[]))])))
        self.assertEqual(session.engines, {})
        self.assertEqual(session.hooks, [])
        self.assertEqual(session.truth.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
