import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flash_v4.observer_runtime import (
    RuntimeObserver,
    install,
    install_megatron_schedule_hook,
)


class ObserverWindowTests(unittest.TestCase):
    def test_profile_stays_inside_requested_complete_iteration_window(self):
        begin, flush = Mock(), Mock()
        observer = RuntimeObserver('profile', library=SimpleNamespace(
            flash_v4_begin_iteration=begin, flash_v4_end_iteration=Mock(), flash_v4_flush=flush))
        for iteration in range(1, 14):
            observer.begin(iteration)
        enabled = [c.args[0] for c in begin.call_args_list if c.args[1]]
        self.assertEqual(enabled, list(range(4, 12)))
        observer.flush(); observer.flush()
        flush.assert_called_once()

    def test_monitor_never_enables_gpu_event_collection(self):
        begin = Mock()
        observer = RuntimeObserver('monitor', library=SimpleNamespace(
            flash_v4_begin_iteration=begin, flash_v4_end_iteration=Mock(), flash_v4_flush=Mock()))
        observer.begin(4); observer.begin(11)
        self.assertTrue(all(call.args[1] == 0 for call in begin.call_args_list))

    def test_missing_preload_and_unbounded_requests_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'preloaded'):
            RuntimeObserver('profile', library=SimpleNamespace())
        with self.assertRaises(ValueError):
            RuntimeObserver('profile', length=9)


class MegatronScheduleHookTests(unittest.TestCase):
    def test_patch_is_installed_after_groups_and_schedule_calls_are_transparent(self):
        events = []
        group_result = object()
        observer = SimpleNamespace(begin=Mock(), end=Mock())
        schedule_calls = []
        factory_calls = []

        def schedule(*, forward_only=False, payload=None):
            schedule_calls.append((forward_only, payload))
            return {'result': payload}

        def factory(*args, **kwargs):
            factory_calls.append((args, kwargs))
            return schedule

        schedules = SimpleNamespace(get_forward_backward_func=factory)

        def initialize_groups(*args, **kwargs):
            events.append(('initialize_groups', args, kwargs))
            return group_result

        adapter = SimpleNamespace(initialize_groups=initialize_groups)

        def import_schedules(module_name):
            events.append(('import_schedules', module_name))
            return schedules

        install_megatron_schedule_hook(adapter, observer, import_schedules)
        self.assertEqual(events, [])
        result = adapter.initialize_groups('output', expert_parallel_size=2)
        self.assertIs(result, group_result)
        self.assertEqual(events[0][0], 'initialize_groups')
        self.assertEqual(events[1], (
            'import_schedules', 'megatron.core.pipeline_parallel.schedules'))

        wrapped_schedule = schedules.get_forward_backward_func('factory-arg', flag=True)
        self.assertEqual(factory_calls, [(('factory-arg',), {'flag': True})])
        self.assertEqual(wrapped_schedule(payload='step-a'), {'result': 'step-a'})
        self.assertEqual(wrapped_schedule(payload='step-b'), {'result': 'step-b'})
        self.assertEqual(schedule_calls, [(False, 'step-a'), (False, 'step-b')])
        self.assertEqual([call.args[0] for call in observer.begin.call_args_list], [1, 2])
        self.assertEqual([call.args[0] for call in observer.end.call_args_list], [1, 2])

    def test_forward_only_schedule_is_rejected_without_counting_or_execution(self):
        observer = SimpleNamespace(begin=Mock(), end=Mock())
        schedule_calls = []

        def schedule(*, forward_only=False):
            schedule_calls.append(forward_only)
            return 'result'

        schedules = SimpleNamespace(get_forward_backward_func=lambda: schedule)
        adapter = SimpleNamespace(initialize_groups=lambda: 'groups-ready')
        install_megatron_schedule_hook(
            adapter, observer, lambda module_name: schedules)
        adapter.initialize_groups()
        wrapped_schedule = schedules.get_forward_backward_func()

        with self.assertRaisesRegex(RuntimeError, 'forward_only evaluation was rejected'):
            wrapped_schedule(forward_only=True)
        observer.begin.assert_not_called()
        self.assertEqual(schedule_calls, [])

    def test_sampling_window_counts_training_schedule_calls_four_through_eleven(self):
        begin, flush = Mock(), Mock()
        observer = RuntimeObserver('profile', library=SimpleNamespace(
            flash_v4_begin_iteration=begin, flash_v4_end_iteration=Mock(), flash_v4_flush=flush))
        schedules = SimpleNamespace(get_forward_backward_func=lambda: (lambda: 'ok'))
        adapter = SimpleNamespace(initialize_groups=lambda: 'groups-ready')
        install_megatron_schedule_hook(adapter, observer, lambda module_name: schedules)
        adapter.initialize_groups()
        schedule = schedules.get_forward_backward_func()
        self.assertEqual([schedule() for _ in range(13)], ['ok'] * 13)
        enabled = [call.args[0] for call in begin.call_args_list if call.args[1]]
        self.assertEqual(enabled, list(range(4, 12)))

    def test_failed_schedule_does_not_mark_iteration_complete(self):
        observer = SimpleNamespace(begin=Mock(), end=Mock())
        def failed_schedule():
            raise RuntimeError('training failure')
        schedules = SimpleNamespace(get_forward_backward_func=lambda: failed_schedule)
        adapter = SimpleNamespace(initialize_groups=lambda: None)
        install_megatron_schedule_hook(adapter, observer, lambda _: schedules)
        adapter.initialize_groups()
        with self.assertRaisesRegex(RuntimeError, 'training failure'):
            schedules.get_forward_backward_func()()
        observer.begin.assert_called_once_with(1)
        observer.end.assert_not_called()

    def test_runtime_install_does_not_replace_or_read_delay_callback(self):
        observer = SimpleNamespace(first=4, last=11, begin=Mock(), end=Mock())
        delay_class = SimpleNamespace(begin_iteration=Mock())
        training_module = SimpleNamespace(GPUStreamDelay=delay_class)
        adapter = SimpleNamespace(initialize_groups=Mock(return_value='ok'))
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'FLASH_V4_OBSERVE_MODE': 'profile',
            'FLASH_V4_PROFILE_START': '4',
            'FLASH_V4_OUTPUT_DIR': temp,
            'RANK': '3',
        }, clear=False), patch('flash_v4.observer_runtime.RuntimeObserver', return_value=observer), \
             patch('flash_v4.observer_runtime.importlib.import_module', return_value=adapter):
            returned = install(training_module, delay_class)
            self.assertIs(returned, observer)
            self.assertIs(training_module.GPUStreamDelay, delay_class)
            delay_class.begin_iteration.assert_not_called()
            setup = json.loads((Path(temp) / 'setup_rank3.json').read_text())
            self.assertEqual(setup['runtime_adapter'], 'Megatron-specific pipeline schedule wrapper')
            self.assertEqual(setup['framework_api'],
                             'megatron.core.pipeline_parallel.schedules.get_forward_backward_func')


if __name__ == '__main__':
    unittest.main()
