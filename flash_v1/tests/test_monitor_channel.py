import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flash_v1.monitor_channel import CAPACITY, MonitorChannel
from flash_v1.online import OnlineSchedule, controller


class MonitorChannelTests(unittest.TestCase):
    def test_two_endpoints_order_wrap_and_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'monitor.bin'
            producer = MonitorChannel.create(path)
            consumer = MonitorChannel(path)
            try:
                for i in range(1, CAPACITY + 1):
                    producer.publish(i, i + .5, 0, i + .25)
                with self.assertRaises(BufferError):
                    producer.publish(CAPACITY + 1, 100, 0, 0)
                rows, cursor = consumer.read_since(0)
                self.assertEqual([row['iteration'] for row in rows], list(range(1, CAPACITY + 1)))
                self.assertEqual(rows[-1]['wall_iteration_ms'], CAPACITY + .5)
                self.assertEqual(rows[-1]['completed_walltime'], CAPACITY + .25)
                producer.publish(CAPACITY + 1, 100, 2, 99)
                rows, cursor = consumer.read_since(cursor)
                self.assertEqual(rows, [dict(iteration=CAPACITY + 1, wall_iteration_ms=100,
                                            profile_epoch=2, completed_walltime=99)])
                with self.assertRaises(BufferError):
                    consumer.read_since(0)
                with self.assertRaises(ValueError):
                    producer.publish(CAPACITY + 3, 100, 0, 0)
            finally:
                producer.close()
                consumer.close()

    def test_mailbox_schedule_is_exact_and_uses_no_redis_or_json_files(self):
        class ForbiddenRedis:
            def __getattr__(self, key):
                raise AssertionError('training schedule accessed Redis')
        with tempfile.TemporaryDirectory() as directory:
            channel = MonitorChannel.create(Path(directory) / 'monitor.bin')
            peer = MonitorChannel(Path(directory) / 'monitor.bin')
            try:
                schedule = OnlineSchedule(ForbiddenRedis(), directory, channel=channel)
                self.assertEqual(schedule.next_epoch(9), 0)
                peer.set_request(1)
                steps = {i + 1: schedule.next_epoch(i) for i in range(10, 20)}
                self.assertEqual([step for step, epoch in steps.items() if epoch], list(range(11, 19)))
                self.assertEqual(peer.get_schedule(), dict(epoch=1, start=11, end=18))
                self.assertEqual([path.name for path in Path(directory).iterdir()], ['monitor.bin'])
            finally:
                channel.close()
                peer.close()

    def test_controller_processes_all_512_even_when_redis_keeps_500(self):
        class Client:
            def __init__(self):
                self.values, self.rows = {}, []

            def exists(self, key):
                return key == 'shutdown' and self.values.get('online_processed_iteration') == CAPACITY

            def set(self, key, value):
                self.values[key] = value

            def pipeline(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def rpush(self, key, *rows):
                self.rows.extend(rows)

            def ltrim(self, key, start, end):
                self.rows = self.rows[start:]

            def execute(self):
                pass
        seen = []
        class Detector:
            def __init__(self, warmup):
                pass

            def add(self, iteration, time_ms):
                seen.append(iteration)
        with tempfile.TemporaryDirectory() as directory:
            channel = MonitorChannel.create(Path(directory) / 'monitor.bin')
            try:
                for i in range(1, CAPACITY + 1):
                    channel.publish(i, 100, 0, i)
                client = Client()
                with patch('flash_v1.online.WindowDetector', Detector), patch('flash_v1.online.time.sleep'):
                    controller(client, directory, {})
                self.assertEqual(seen, list(range(1, CAPACITY + 1)))
                self.assertEqual(len(client.rows), 500)
                self.assertEqual(json.loads(client.rows[-1])['iteration'], CAPACITY)
                self.assertEqual(client.values['online_latest_iteration'], CAPACITY)
            finally:
                channel.close()


if __name__ == '__main__':
    unittest.main()
