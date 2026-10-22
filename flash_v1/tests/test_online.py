import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flash_v1.online import OnlineSchedule, controller, drains_after, missing_coverage, window_rows


class Client:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value


class OnlineTests(unittest.TestCase):
    def test_exactly_eight_steps_and_next_request(self):
        client = Client()
        with tempfile.TemporaryDirectory() as output:
            schedule = OnlineSchedule(client, output)
            self.assertEqual(schedule.next_epoch(9), 0)
            client.set('online_request', json.dumps(dict(epoch=1)))
            epochs = {i + 1: schedule.next_epoch(i) for i in range(10, 20)}
            self.assertEqual([i for i, epoch in epochs.items() if epoch], list(range(11, 19)))
            self.assertEqual(json.loads(client.get('online_schedule_1')), dict(epoch=1, start=11, end=18))
            self.assertEqual(len((Path(output) / 'events.jsonl').read_text().splitlines()), 1)
            client.set('online_request', json.dumps(dict(epoch=2)))
            self.assertEqual(schedule.next_epoch(20), 2)

    def test_coverage_filters_boundaries_and_requires_all_ranks(self):
        schedule = dict(start=11, end=18)
        rows = [dict(iteration=i, rank=r) for i in range(10, 20) for r in range(16)]
        filtered = window_rows(rows, schedule)
        self.assertEqual(len(filtered), 128)
        self.assertEqual(missing_coverage(filtered, schedule), [])
        self.assertEqual(missing_coverage(filtered[:-1], schedule), [(18, 15)])

    def test_both_drains_must_follow_completion(self):
        client = Client()
        client.set('online_drained_node0', json.dumps(dict(through_iteration=101)))
        self.assertFalse(drains_after(client, 100))
        client.set('online_drained_node1', json.dumps(dict(through_iteration=99)))
        self.assertFalse(drains_after(client, 100))
        client.set('online_drained_node1', json.dumps(dict(through_iteration=100)))
        self.assertTrue(drains_after(client, 100))

    def test_controller_excludes_profile_resets_and_respects_quota(self):
        class Training(Client):
            polls = 0
            def exists(self, key):
                return key == 'shutdown' and self.polls >= 3

            def set(self, key, value):
                super().set(key, value)
                if key == 'online_request':
                    self.values['online_schedule_1'] = json.dumps(dict(epoch=1, start=2, end=9))

            def lrange(self, key, start, end):
                if start == -1:
                    return [json.dumps(dict(iteration=10))]
                self.polls += 1
                last = 1 if self.polls == 1 else 10 if self.polls == 2 else 11
                return [json.dumps(dict(iteration=i, wall_iteration_ms=100,
                                        profile_epoch=1 if 2 <= i <= 9 else 0))
                        for i in range(1, last + 1)]
        seen, warmups = [], []
        class Detector:
            def __init__(self, warmup):
                warmups.append(warmup)

            def add(self, iteration, duration):
                seen.append(iteration)
                return dict(iteration=iteration)
        client = Training()
        with tempfile.TemporaryDirectory() as output, \
                patch('flash_v1.online.WindowDetector', Detector), \
                patch('flash_v1.online.drains_after', return_value=True), \
                patch('flash_v1.online.diagnose', return_value=dict(predicted='healthy')), \
                patch('flash_v1.online.time.sleep'):
            controller(client, output, {}, warmup=12, max_profiles=1)
            events = [json.loads(line) for line in (Path(output) / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(seen, [1])
        self.assertEqual(warmups, [12, 10])
        self.assertEqual([row['event'] for row in events], ['detection', 'localization', 'quota_exhausted'])
        self.assertEqual(client.get('online_pending'), '0')
        self.assertIsNotNone(client.get('online_completed_1'))

    def test_training_end_handles_missing_truncated_and_late_complete_schedule(self):
        for schedule, expected, calls in [
                (None, 'training_ended_before_profile_start', 0),
                (dict(epoch=1, start=3, end=10), 'truncated_profile', 0),
                (dict(epoch=1, start=2, end=9), None, 1)]:
            with self.subTest(schedule=schedule):
                class FinishedTraining(Client):
                    schedule_reads = 0
                    def exists(self, key):
                        return key == 'shutdown' and 'online_completed_1' in self.values

                    def get(self, key):
                        if key == 'online_training_done':
                            return json.dumps(dict(last_iteration=9))
                        if key == 'online_schedule_1':
                            self.schedule_reads += 1
                            # Only expose the schedule after every iteration
                            # row has already been consumed in the first batch.
                            if schedule and self.schedule_reads >= 9:
                                return json.dumps(schedule)
                            return None
                        return super().get(key)

                    def lrange(self, key, start, end):
                        if start == -1:
                            return [json.dumps(dict(iteration=9))]
                        return [json.dumps(dict(iteration=i, wall_iteration_ms=100, profile_epoch=0))
                                for i in range(1, 10)]
                class Detector:
                    def __init__(self, warmup):
                        pass

                    def add(self, iteration, duration):
                        return dict(iteration=iteration, profile_seconds=20)
                client = FinishedTraining()
                with tempfile.TemporaryDirectory() as output, \
                        patch('flash_v1.online.WindowDetector', Detector), \
                        patch('flash_v1.online.drains_after', return_value=True), \
                        patch('flash_v1.online.diagnose', return_value=dict(predicted='healthy')) as diagnose, \
                        patch('flash_v1.online.time.sleep'):
                    controller(client, output, {}, max_profiles=1)
                    events = [json.loads(line) for line in (Path(output) / 'events.jsonl').read_text().splitlines()]
                result = json.loads(client.get('online_completed_1'))
                self.assertEqual(result.get('reason'), expected)
                self.assertEqual(diagnose.call_count, calls)
                self.assertEqual(client.get('online_pending'), '0')
                self.assertEqual(client.get('online_processed_iteration'), 9)
                self.assertEqual(events[0]['profile_iterations'], 8)
                self.assertNotIn('profile_seconds', events[0])
                self.assertEqual(events[-1]['event'], 'localization')


if __name__ == '__main__':
    unittest.main()
