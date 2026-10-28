import json
import tempfile
import unittest
from pathlib import Path

from flash_v4.bocd_monitor import (
    JsonlBuffer,
    REQUEST_METHOD,
    REQUEST_SCHEMA,
    consume_rows,
    run_worker,
)


class FakeDetector:
    def __init__(self, trigger_iteration=None, failure=None):
        self.trigger_iteration = trigger_iteration
        self.failure = failure
        self.last_iteration = None
        self.calls = []

    def add(self, iteration, time_ms):
        if self.failure is not None:
            raise self.failure
        if self.last_iteration is not None and iteration <= self.last_iteration:
            raise ValueError("iterations must strictly increase")
        self.last_iteration = iteration
        self.calls.append((iteration, time_ms))
        if iteration == self.trigger_iteration:
            return {"iteration": iteration, "method": "Rbeast", "value": time_ms}
        return None


class BOCdMonitorTests(unittest.TestCase):
    def test_consume_rows_passes_only_timing_fields_and_returns_count(self):
        detector = FakeDetector()
        event, count = consume_rows(
            [
                {"iteration": 1, "time_ms": 20.0, "unused": "ignored"},
                {"iteration": 2, "time_ms": 21.0, "rank": 0},
            ],
            detector,
        )
        self.assertIsNone(event)
        self.assertEqual(count, 2)
        self.assertEqual(detector.calls, [(1, 20.0), (2, 21.0)])

    def test_duplicate_iteration_propagates_detector_error(self):
        detector = FakeDetector()
        rows = [
            {"iteration": 4, "time_ms": 20.0},
            {"iteration": 4, "time_ms": 20.0},
        ]
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            consume_rows(rows, detector)
        self.assertEqual(detector.calls, [(4, 20.0)])

    def test_trigger_returns_first_event_and_stops_consuming_batch(self):
        detector = FakeDetector(trigger_iteration=2)
        event, count = consume_rows(
            [
                {"iteration": 1, "time_ms": 20.0},
                {"iteration": 2, "time_ms": 30.0},
                {"iteration": 3, "time_ms": 40.0},
            ],
            detector,
        )
        self.assertEqual(event["iteration"], 2)
        self.assertEqual(count, 2)
        self.assertEqual(detector.calls, [(1, 20.0), (2, 30.0)])

    def test_jsonl_buffer_holds_partial_line_until_newline(self):
        buffer = JsonlBuffer()
        self.assertEqual(buffer.feed('{"iteration":1,"time_ms":2'), [])
        complete = buffer.feed('0}\n{"iteration":2,"time_ms":3}\n{"iteration":3,"time_ms":3')
        self.assertEqual(complete, ['{"iteration":1,"time_ms":20}', '{"iteration":2,"time_ms":3}'])
        self.assertEqual(buffer.feed('0}\n'), ['{"iteration":3,"time_ms":30}'])

    def test_untriggered_worker_summarizes_and_exits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "train.jsonl"
            log.write_text(
                '{"iteration":1,"time_ms":20,"extra":"ignored"}\n'
                '{"iteration":2,"time_ms":21}\n',
                encoding="utf-8",
            )
            detector = FakeDetector()
            code = run_worker(
                str(log), str(root / "request.json"), str(root / "audit.json"),
                str(root / "stop"), max_seconds=0.01, poll_seconds=0.001,
                detector=detector,
            )
            audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertFalse((root / "request.json").exists())
            self.assertEqual(audit["count"], 2)
            self.assertFalse(audit["event"])
            self.assertIsNone(audit["error"])
            self.assertEqual(set(audit), {"start", "end", "event", "error", "count", "finish_reason"})
            self.assertEqual(audit['finish_reason'], 'budget')

    def test_trigger_publishes_unique_request_and_stops(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "train.jsonl"
            log.write_text(
                '{"iteration":1,"time_ms":20}\n'
                '{"iteration":2,"time_ms":30}\n'
                '{"iteration":3,"time_ms":40}\n',
                encoding="utf-8",
            )
            detector = FakeDetector(trigger_iteration=2)
            request_path = root / "request.json"
            code = run_worker(
                str(log), str(request_path), str(root / "audit.json"),
                str(root / "stop"), max_seconds=1, poll_seconds=0.001,
                detector=detector,
            )
            request = json.loads(request_path.read_text(encoding="utf-8"))
            audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(request["schema"], REQUEST_SCHEMA)
            self.assertEqual(request["iteration"], 2)
            self.assertEqual(request["method"], REQUEST_METHOD)
            self.assertEqual(request["event"]["iteration"], 2)
            self.assertEqual(detector.calls, [(1, 20), (2, 30)])
            self.assertTrue(audit["event"])
            self.assertEqual(audit["count"], 2)

    def test_worker_records_detector_exception_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "train.jsonl"
            log.write_text('{"iteration":1,"time_ms":20}\n', encoding="utf-8")
            audit_path = root / "audit.json"
            code = run_worker(
                str(log), str(root / "request.json"), str(audit_path),
                str(root / "stop"), max_seconds=1, poll_seconds=0.001,
                detector=FakeDetector(failure=RuntimeError("test detector failure")),
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertIn("RuntimeError: test detector failure", audit["error"])
            self.assertEqual(audit["count"], 0)
            self.assertFalse((root / "request.json").exists())


if __name__ == "__main__":
    unittest.main()
