"""Opt-in, bounded schedule-completion receipts for an external clock watcher.

This is not a detector input. Linux containers and their host share CLOCK_MONOTONIC.
Every rank publishes the same schema, without actuator targets or credentials.
"""
import json
import math
from pathlib import Path
import re
import time

CONTRACT = 'r66_rank_schedule_v1'
MAX_ITERATIONS = 512
MAX_BYTES = 1024


def validate_receipt(row, *, run_id, rank, now, previous=0, max_age=3):
    if not isinstance(row, dict):
        raise ValueError('invalid progress schema')
    expected = dict(contract=CONTRACT, run_id=run_id, rank=rank,
                    node_rank=rank // 8, local_rank=rank % 8, status='running')
    if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
        raise ValueError('progress identity or status mismatch')
    iteration, stamp = row.get('iteration'), row.get('monotonic_seconds')
    if type(iteration) is not int or not previous <= iteration <= MAX_ITERATIONS or iteration < 1:
        raise ValueError('invalid or nonmonotonic progress iteration')
    if (type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not 0 <= now - stamp <= max_age):
        raise ValueError('stale or future progress timestamp')
    return iteration


class ProgressObserver:
    def __init__(self, delegate, directory, rank, local_rank, run_id, clock=None):
        if (type(rank) is not int or not 0 <= rank < 16 or local_rank != rank % 8
                or not isinstance(run_id, str) or not re.fullmatch(r'v4_[a-f0-9]{32}', run_id)):
            raise ValueError('invalid rank progress identity')
        self.delegate, self.rank, self.run_id = delegate, rank, run_id
        self.clock = clock or time.monotonic
        self.iteration, self.closed = 0, False
        root = Path(directory) / 'progress'
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f'rank{rank}.json'
        self.temp = self.path.with_suffix('.tmp')
        if self.path.exists() or self.temp.exists():
            raise FileExistsError('progress receipt already exists; no reuse')

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def begin(self, iteration):
        if self.closed or iteration != self.iteration + 1 or iteration > MAX_ITERATIONS:
            raise ValueError('progress schedule exceeds bounds or is nonmonotonic')
        self.delegate.begin(iteration)

    def publish(self, status):
        row = dict(contract=CONTRACT, run_id=self.run_id, rank=self.rank,
                   node_rank=self.rank // 8, local_rank=self.rank % 8,
                   status=status, iteration=self.iteration, monotonic_seconds=self.clock())
        raw = json.dumps(row)
        if len(raw.encode()) > MAX_BYTES:
            raise ValueError('oversized progress receipt')
        try:
            with self.temp.open('x') as stream:
                stream.write(raw)
            self.temp.replace(self.path)
        finally:
            self.temp.unlink(missing_ok=True)

    def end(self, iteration):
        if self.closed or iteration != self.iteration + 1 or iteration > MAX_ITERATIONS:
            raise ValueError('progress schedule completion mismatch')
        self.delegate.end(iteration)
        self.iteration = iteration
        self.publish('running')

    def flush(self):
        if self.closed:
            return
        try:
            self.delegate.flush()
        finally:
            self.closed = True
            self.publish('closed')
