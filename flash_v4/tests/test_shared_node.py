import unittest
from unittest.mock import patch
from flash_v4.shared_node import (
    check_name,
    cleanup_private_service,
    terminate_private_clients,
    physical_uuid_order,
    prepare_training_environment,
)
import tempfile
from pathlib import Path


class PrivateMpsTests(unittest.TestCase):
    def test_pre_service_failure_confirms_nothing_started_without_quit(self):
        with patch('flash_v4.shared_node.control') as command:
            self.assertEqual(cleanup_private_service({}, None, False, False), (True, None, None))
            command.assert_not_called()

    def test_rejects_nonprivate_or_path_run_names(self):
        for name in ('default', '../other', 'v4_run;quit', 'v4_', 'v4_' + 'x' * 61):
            with self.assertRaises(ValueError):
                check_name(name)

    def test_cleanup_targets_only_clients_returned_by_private_daemon(self):
        env = {'CUDA_MPS_PIPE_DIRECTORY': '/tmp/v4_example_mps_pipe'}
        replies = {'get_server_list': 'header\n41\n',
                   'get_client_list 41': '51\n52\nnot-a-pid',
                   'terminate_client 41 51': '', 'terminate_client 41 52': ''}
        with patch('flash_v4.shared_node.control', side_effect=lambda e, c: replies[c]) as ctl:
            terminate_private_clients(env)
        commands = [call.args[1] for call in ctl.call_args_list]
        self.assertEqual(commands, ['get_server_list', 'get_client_list 41',
                                    'terminate_client 41 51', 'terminate_client 41 52'])
        self.assertTrue(all(call.args[0] is env for call in ctl.call_args_list))

    def test_physical_uuid_binding_rejects_duplicate_or_missing_devices(self):
        identifiers = ['GPU-00000000-0000-0000-0000-' + str(i).zfill(12) for i in range(8)]
        rows = [str(i) + ', ' + identifiers[i] for i in range(8)]
        self.assertEqual(physical_uuid_order('\n'.join(reversed(rows))), identifiers)
        with self.assertRaises(ValueError):
            physical_uuid_order('\n'.join(rows[:-1]))
        with self.assertRaises(ValueError):
            physical_uuid_order('\n'.join(rows[:-1] + [rows[0]]))

    def test_empty_private_daemon_does_not_issue_any_termination(self):
        with patch('flash_v4.shared_node.control', return_value='') as ctl:
            terminate_private_clients({'CUDA_MPS_PIPE_DIRECTORY': '/tmp/v4_only'})
        ctl.assert_called_once()

    def test_nonzero_training_exit_terminates_private_clients_before_private_quit(self):
        events = []
        env = {'CUDA_MPS_PIPE_DIRECTORY': '/tmp/v4_private_pipe'}
        proc = type('Proc', (), {'returncode': 17})()

        def terminate(received_env):
            self.assertIs(received_env, env)
            events.append('terminate_private_clients')

        def control(received_env, command):
            self.assertIs(received_env, env)
            events.append(command)
            return ''

        with patch('flash_v4.shared_node.terminate_private_clients', side_effect=terminate), \
             patch('flash_v4.shared_node.control', side_effect=control):
            result = cleanup_private_service(env, proc, True, True)
        self.assertEqual(events, ['terminate_private_clients', 'quit'])
        self.assertEqual(result, (True, None, None))

    def test_successful_exit_skips_client_termination_and_reaps_private_service(self):
        events = []
        env = {'CUDA_MPS_PIPE_DIRECTORY': '/tmp/v4_private_pipe'}
        proc = type('Proc', (), {'returncode': 0})()
        with patch('flash_v4.shared_node.terminate_private_clients',
                   side_effect=lambda received_env: events.append('clients')), \
             patch('flash_v4.shared_node.control',
                   side_effect=lambda received_env, command: events.append(command)):
            result = cleanup_private_service(env, proc, True, True)
        self.assertEqual(events, ['quit'])
        self.assertEqual(result, (True, None, None))

    def test_private_quit_is_still_attempted_if_client_cleanup_fails(self):
        events = []
        env = {'CUDA_MPS_PIPE_DIRECTORY': '/tmp/v4_private_pipe'}
        proc = type('Proc', (), {'returncode': 17})()

        def fail_client_cleanup(received_env):
            self.assertIs(received_env, env)
            events.append('terminate_private_clients')
            raise RuntimeError('simulated private cleanup failure')

        def control(received_env, command):
            self.assertIs(received_env, env)
            events.append(command)
            return ''

        with patch('flash_v4.shared_node.terminate_private_clients',
                   side_effect=fail_client_cleanup), \
             patch('flash_v4.shared_node.control', side_effect=control):
            result = cleanup_private_service(env, proc, True, True)
        self.assertEqual(events, ['terminate_private_clients', 'quit'])
        self.assertEqual(result, (True, 'RuntimeError', None))

    def test_observer_preload_precedes_actor_for_full_call_envelope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actor = root / 'libstream.so'
            observer = root / 'libobserver.so'
            actor.touch()
            observer.touch()
            env = {
                'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': '17',
                'FLASH_V4_QUOTA_PRIVATE': '{"3": 100}',
                'FLASH_V4_STREAM_DELAY_PRIVATE': '{"3": 0.5}',
                'KEPT': 'yes',
            }
            controller, training, enabled = prepare_training_environment(
                env, 'monitor', observer, actor)

        self.assertTrue(enabled)
        self.assertEqual(controller['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'], '100')
        self.assertNotIn('FLASH_V4_QUOTA_PRIVATE', controller)
        self.assertNotIn('FLASH_V4_STREAM_DELAY_PRIVATE', controller)
        self.assertEqual(training['FLASH_V4_STREAM_DELAY_PRIVATE'], '{"3": 0.5}')
        self.assertEqual(training['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'], '100')
        preloads = training['LD_PRELOAD'].split(':')
        self.assertTrue(preloads[0].endswith('libobserver.so'))
        self.assertTrue(preloads[1].endswith('libstream.so'))
        self.assertNotIn('LD_PRELOAD', controller)

    def test_empty_ratio_map_enables_all_rank_sham_and_actor_absence_stays_off(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / 'libstream.so'
            library.touch()
            sham_env = {'FLASH_V4_STREAM_DELAY_PRIVATE': '{}'}
            _, sham_training, enabled = prepare_training_environment(
                sham_env, stream_delay_library=library)
            self.assertTrue(enabled)
            self.assertEqual(sham_training['FLASH_V4_STREAM_DELAY_PRIVATE'], '{}')
            self.assertEqual(sham_training['LD_PRELOAD'], str(library.resolve()))

        env = {}
        _, training, enabled = prepare_training_environment(env)
        self.assertFalse(enabled)
        self.assertNotIn('LD_PRELOAD', training)
        self.assertEqual(training['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'], '100')

    def test_shared_actor_environment_rejects_low_mps_quota_or_missing_library(self):
        with self.assertRaisesRegex(ValueError, 'quota 100'):
            prepare_training_environment({'FLASH_V4_QUOTA_PRIVATE': '{"3": 50}'})
        with self.assertRaisesRegex(ValueError, 'existing native library'):
            prepare_training_environment(
                {'FLASH_V4_STREAM_DELAY_PRIVATE': '{}'},
                stream_delay_library=Path('/not/a/library.so'))


if __name__ == '__main__':
    unittest.main()
