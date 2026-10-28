import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from flash_v4.online_runtime import OnlineObserver, ProfileState, child_environment


class ProfileControlTests(unittest.TestCase):
    def test_same_eight_iteration_window_and_permanent_cooldown(self):
        states = [ProfileState() for _ in range(16)]
        enabled = [[] for _ in states]
        for iteration in range(1, 65):
            for rank, state in enumerate(states):
                if iteration == 36:
                    self.assertTrue(state.check_due(iteration))
                    state.start(iteration)
                if state.begin(iteration):
                    enabled[rank].append(iteration)
                state.end(iteration)
        self.assertTrue(all(v == list(range(36, 44)) for v in enabled))
        self.assertTrue(all(s.complete and not s.check_due(60) for s in states))
        with self.assertRaises(RuntimeError):
            states[0].start(68)

    def test_skipped_or_repeated_iteration_fails_closed(self):
        state = ProfileState()
        with self.assertRaises(RuntimeError):
            state.begin(2)
        state.begin(1)
        with self.assertRaises(RuntimeError):
            state.begin(1)

    def test_cpu_consumer_never_inherits_gpu_probe_or_private_payload(self):
        result = child_environment({'LD_PRELOAD': '/probe.so', 'CUDA_VISIBLE_DEVICES': '0',
                                    'FLASH_V4_STREAM_DELAY_PRIVATE': 'secret', 'PYTHONPATH': '/code'})
        self.assertNotIn('LD_PRELOAD', result)
        self.assertNotIn('FLASH_V4_STREAM_DELAY_PRIVATE', result)
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(result['PYTHONPATH'], '/code')

    def test_ready_marker_follows_native_flush_and_no_monitor_cuda_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Mock()
            obs = OnlineObserver(native, directory, rank=1)
            obs.group = object()
            obs.dist = Mock()
            obs.payload = [0]
            for iteration in range(1, 4):
                obs.begin(iteration); obs.end(iteration)
            native.begin_fn.assert_not_called()
            obs.state.start(4)
            for iteration in range(4, 12):
                obs.begin(iteration); obs.end(iteration)
            self.assertEqual(native.begin_fn.call_count, 8)
            native.flush.assert_called_once()
            self.assertTrue((Path(directory) / 'ready_rank1.json').exists())
            obs.flush()

    def test_invalid_request_is_broadcast_as_error_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            obs = OnlineObserver(Mock(), directory, rank=0)
            obs.group, obs.dist, obs.payload = object(), Mock(), [0]
            (Path(directory) / 'profile_request.json').write_text('{broken')
            for iteration in range(1, 4):
                obs.begin(iteration); obs.end(iteration)
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                obs.begin(4)
            self.assertEqual(obs.payload[0], -1)
            obs.dist.broadcast.assert_called_once()
            obs.flush()


if __name__ == '__main__':
    unittest.main()
