"""External CPU worker for the FLASH v1 Rbeast monitoring detector."""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from flash_v1.analysis import WindowDetector


REQUEST_SCHEMA = "flash_v4.monitor_request.v1"
REQUEST_METHOD = "flash_v1_Rbeast"


def consume_rows(rows: Iterable[Mapping[str, Any]], detector: Any) -> Tuple[Optional[Any], int]:
    """Feed parsed log rows to a detector, passing only iteration and time_ms.

    Returns the first event and number of successfully consumed rows. Detector
    exceptions, including duplicate-iteration errors, propagate to the caller.
    """
    count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("training log row must be a JSON object")
        try:
            iteration = row["iteration"]
            time_ms = row["time_ms"]
        except KeyError as exc:
            raise ValueError("training log row requires iteration and time_ms") from exc
        event = detector.add(iteration, time_ms)
        count += 1
        if event is not None:
            return event, count
    return None, count


class JsonlBuffer:
    """Split appended text into complete newline-terminated JSONL records."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> List[str]:
        if not isinstance(chunk, str):
            raise TypeError("JSONL chunks must be text")
        pieces = (self._pending + chunk).split("\n")
        self._pending = pieces.pop()
        return [line for line in pieces if line.strip()]


def _parse_training_row(line: str) -> Dict[str, Any]:
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("training log line must contain a JSON object")
    try:
        return {"iteration": row["iteration"], "time_ms": row["time_ms"]}
    except KeyError as exc:
        raise ValueError("training log row requires iteration and time_ms") from exc


def _atomic_create_json(path: str, payload: Mapping[str, Any]) -> None:
    """Publish one complete JSON document without replacing an existing path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, str(target))
        except FileExistsError as exc:
            raise FileExistsError("output path already exists: " + str(target)) from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_worker(
    training_log: str,
    request_path: str,
    audit_path: str,
    stop_path: str,
    max_seconds: float = 150.0,
    poll_seconds: float = 0.2,
    detector: Any = None,
    degradation: float = 0.2,
) -> int:
    """Tail rank0's JSONL log, publish the first event, and write a summary audit."""
    started = _utc_now()
    started_mono = time.monotonic()
    count = 0
    event_found = False
    error: Optional[str] = None
    exit_code = 0
    finish_reason = 'budget'

    try:
        if max_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("max_seconds and poll_seconds must be positive")
        if not math.isfinite(degradation) or not 0 < degradation < 1:
            raise ValueError("degradation must be finite and between 0 and 1")
        if Path(request_path).absolute() == Path(audit_path).absolute():
            raise ValueError("request and audit paths must be different")
        active_detector = detector if detector is not None else WindowDetector(degradation=degradation)
        buffer = JsonlBuffer()
        deadline = started_mono + max_seconds
        with open(training_log, "r", encoding="utf-8") as source:
            done = False
            while not done:
                if os.path.exists(stop_path):
                    finish_reason = 'stop'
                    break
                for line in buffer.feed(source.read()):
                    if os.path.exists(stop_path) or time.monotonic() >= deadline:
                        finish_reason = 'stop' if os.path.exists(stop_path) else 'budget'
                        done = True
                        break
                    row = _parse_training_row(line)
                    event, consumed = consume_rows((row,), active_detector)
                    count += consumed
                    if event is not None:
                        event_found = True
                        finish_reason = 'request'
                        request = {
                            "schema": REQUEST_SCHEMA,
                            "event": event,
                            "iteration": row["iteration"],
                            "method": REQUEST_METHOD,
                            "degradation_threshold": degradation,
                        }
                        _atomic_create_json(request_path, request)
                        done = True
                        break
                if done:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_seconds, remaining))
    except Exception as exc:
        finish_reason = 'error'
        error = f"{type(exc).__name__}: {exc}"
        exit_code = 1

    audit = {
        "start": started,
        "end": _utc_now(),
        "event": event_found,
        "error": error,
        "count": count,
        "finish_reason": finish_reason,
    }
    try:
        _atomic_create_json(audit_path, audit)
    except Exception as exc:
        print(f"audit write failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if error is not None:
        print(error, file=sys.stderr)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True, help="rank0 train.jsonl path")
    parser.add_argument("--request", required=True, help="new atomic request JSON path")
    parser.add_argument("--audit", required=True, help="new summary audit JSON path")
    parser.add_argument("--stop", required=True, help="stop when this file exists")
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--degradation", type=float, default=0.2)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_worker(
        training_log=args.training_log,
        request_path=args.request,
        audit_path=args.audit,
        stop_path=args.stop,
        max_seconds=args.max_seconds,
        poll_seconds=args.poll_seconds,
        degradation=args.degradation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
