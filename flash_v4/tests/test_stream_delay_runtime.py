"""CPU-only contracts for the Megatron-specific stream-delay adapter."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flash_v4 import stream_delay_runtime


class FakeFunction:
    def __init__(self, name, result=None, calls=None):
        self.name = name
        self.result = result
        self.calls = calls if calls is not None else []

    def __call__(self, *args):
        self.calls.append((self.name, *args))
        return self.result


class FakeNativeLibrary:
    def __init__(self, calls, configure_status=0):
        self.flash_v4_delay_configure = FakeFunction(
            "configure", configure_status, calls)
        self.flash_v4_delay_begin = FakeFunction("begin", None, calls)
        self.flash_v4_delay_phase = FakeFunction("phase", None, calls)
        self.flash_v4_delay_end = FakeFunction("end", None, calls)
        self.flash_v4_delay_flush = FakeFunction("flush", None, calls)


class NativeBindingTests(unittest.TestCase):
    def test_zero_ratio_sham_uses_native_config_and_warmup_fault_window(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private.jsonl"
            actor = stream_delay_runtime.NativeStreamDelayActor(
                0.0, path, FakeNativeLibrary(calls))
            for iteration in range(1, 13):
                actor.begin(iteration)
                actor.end(iteration)
            actor.flush()

        self.assertEqual(calls[0], ("configure", 0.0, str(path).encode()))
        begins = [call for call in calls if call[0] == "begin"]
        self.assertEqual(begins[:4], [
            ("begin", 1, 1, 0),
            ("begin", 2, 1, 0),
            ("begin", 3, 1, 0),
            ("begin", 4, 0, 1),
        ])
        self.assertEqual(begins[-4], ("begin", 9, 0, 1))
        self.assertEqual(begins[-3], ("begin", 10, 0, 1))
        self.assertEqual(begins[-2], ("begin", 11, 0, 1))
        self.assertEqual(begins[-1], ("begin", 12, 0, 0))
        self.assertEqual(sum(call[0] == "flush" for call in calls), 1)

    def test_delayed_bounded_window_keeps_calibration_and_faults_40_through_71(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            actor = stream_delay_runtime.NativeStreamDelayActor(
                0.5, Path(temp) / "private.jsonl", FakeNativeLibrary(calls),
                first=40, last=71)
            for iteration in range(1, 73):
                actor.begin(iteration)
                actor.end(iteration)
            actor.flush()

        begins = [call for call in calls if call[0] == "begin"]
        self.assertEqual([call[1] for call in begins if call[2]], [1, 2, 3])
        self.assertEqual([call[1] for call in begins if call[3]], list(range(40, 72)))
        self.assertEqual(begins[-1], ("begin", 72, 0, 0))

    def test_install_uses_startup_window_and_defaults_for_older_startup_objects(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            startup = SimpleNamespace(
                stream_delay_enabled=True,
                requested_quota_percent=100,
                actuation_dir=Path(temp),
                stream_delay_ratio=0.5,
                stream_delay_first=40,
                stream_delay_last=71,
                rank=3,
            )
            installed = []
            legacy_startup = SimpleNamespace(
                stream_delay_enabled=True,
                requested_quota_percent=100,
                actuation_dir=Path(temp),
                stream_delay_ratio=0.0,
                rank=4,
            )
            with patch.object(stream_delay_runtime.importlib, "import_module",
                              return_value=SimpleNamespace()), \
                 patch.object(stream_delay_runtime, "_assert_cuda_not_initialized"), \
                 patch.object(stream_delay_runtime, "install_megatron_schedule_hook",
                              side_effect=lambda adapter, actor: installed.append(actor)):
                actor = stream_delay_runtime.install(
                    SimpleNamespace(), startup, FakeNativeLibrary(calls))
                legacy_calls = []
                legacy = stream_delay_runtime.install(
                    SimpleNamespace(), legacy_startup, FakeNativeLibrary(legacy_calls))
            self.assertEqual((actor.first, actor.last), (40, 71))
            self.assertEqual((legacy.first, legacy.last), (4, 11))
            self.assertEqual(installed, [actor, legacy])

    def test_native_configure_failure_is_reported_without_enabling(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "configure failed \\(-2\\)"):
                stream_delay_runtime.NativeStreamDelayActor(
                    0.5, Path(temp) / "private.jsonl",
                    FakeNativeLibrary(calls, configure_status=-2))
        self.assertEqual(len(calls), 1)


class ScheduleHookTests(unittest.TestCase):
    def make_schedule_module(self):
        module = SimpleNamespace()

        def forward_step():
            return "forward"

        def backward_step():
            return "backward"

        module.forward_step = forward_step
        module.backward_step = backward_step

        def schedule(forward_only=False):
            if forward_only:
                return "eval"
            module.forward_step()
            module.backward_step()
            return "trained"

        module.get_forward_backward_func = lambda: schedule
        return module

    def test_actor_composes_outside_existing_observer_schedule_adapter(self):
        events = []
        schedules = self.make_schedule_module()
        adapter = SimpleNamespace(initialize_groups=lambda: events.append("groups"))

        class Observer:
            def begin(self, iteration):
                events.append(("observer_begin", iteration))

            def end(self, iteration):
                events.append(("observer_end", iteration))

        class Actor:
            def begin(self, iteration):
                events.append(("actor_begin", iteration))

            def phase(self, phase):
                events.append(("phase", phase))

            def end(self, iteration):
                events.append(("actor_end", iteration))

        from flash_v4.observer_runtime import install_megatron_schedule_hook

        install_megatron_schedule_hook(
            adapter, Observer(), schedule_importer=lambda _: schedules)
        stream_delay_runtime.install_megatron_schedule_hook(
            adapter, Actor(), schedule_importer=lambda _: schedules)
        adapter.initialize_groups()
        result = schedules.get_forward_backward_func()()

        self.assertEqual(result, "trained")
        self.assertLess(events.index(("actor_begin", 1)),
                        events.index(("observer_begin", 1)))
        self.assertLess(events.index(("observer_end", 1)),
                        events.index(("actor_end", 1)))
        self.assertEqual([event for event in events if isinstance(event, tuple)
                          and event[0] == "phase"], [
                              ("phase", 1), ("phase", 0),
                              ("phase", 2), ("phase", 0),
                          ])

    def test_forward_only_is_rejected_before_actor_iteration_counter(self):
        events = []
        schedules = self.make_schedule_module()
        adapter = SimpleNamespace(initialize_groups=lambda: None)
        actor = SimpleNamespace(
            begin=lambda iteration: events.append(("begin", iteration)),
            phase=lambda phase: events.append(("phase", phase)),
            end=lambda iteration: events.append(("end", iteration)),
        )
        stream_delay_runtime.install_megatron_schedule_hook(
            adapter, actor, schedule_importer=lambda _: schedules)
        adapter.initialize_groups()
        schedule = schedules.get_forward_backward_func()
        with self.assertRaisesRegex(RuntimeError, "forward_only evaluation"):
            schedule(forward_only=True)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
