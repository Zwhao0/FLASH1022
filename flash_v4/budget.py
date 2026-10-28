"""CPU-side collector primitives; no CUDA operations or training integration."""
import hashlib
import json
import math

# Deliberately excludes truth, requested/actual frequency, power limits, and
# injection times. Member IDs must be real observed communicator members.
ALLOWED = frozenset(('kind', 'rank', 'iteration', 'sequence', 'cpu_ns',
                     'stream', 'comm_id', 'members', 'op', 'numel', 'dtype',
                     'shape', 'strides', 'elapsed_us', 'overlap_us', 'count'))


def encode_observation(row):
    if set(row) - ALLOWED:
        raise ValueError('unexpected fields in detector observations')
    if row.get('kind') not in ('progress', 'communication', 'computation'):
        raise ValueError('unknown observation kind')
    for key in ('rank', 'iteration'):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f'invalid {key}')
    if row['rank'] >= 16:
        raise ValueError('rank outside declared world')
    for key in ('elapsed_us', 'overlap_us'):
        value = row.get(key)
        if value is not None and (isinstance(value, bool) or
                not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
            raise ValueError('invalid timing')
    # Scalar or flat scalar-array fields only: nested payloads cannot conceal
    # injected labels inside an otherwise permitted field.
    for value in row.values():
        values = value if isinstance(value, (list, tuple)) else [value]
        if any(type(v) not in (str, int, float, bool, type(None)) for v in values):
            raise ValueError('nested observation payload')
        if any(isinstance(v, float) and not math.isfinite(v) for v in values):
            raise ValueError('non-finite observation payload')
    data = (json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(data) > 1024:
        raise ValueError('event exceeds fixed serialization budget')
    return data


def selected(comparable_key, stride=16):
    """A rank-independent key enables paired sampling; adapter must align keys."""
    if type(stride) is not int or stride < 1:
        raise ValueError('invalid sampling stride')
    digest = hashlib.blake2b(comparable_key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'little') % stride == 0


class ProfileBudget:
    def __init__(self, first_iteration, start_seconds, *, iterations=8,
                 seconds=30., byte_limit=1_048_576):
        if (type(first_iteration) is not int or first_iteration < 0 or
                type(iterations) is not int or iterations < 1 or
                type(byte_limit) is not int or byte_limit < 1 or
                not math.isfinite(start_seconds) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('invalid profile budget')
        self.first = first_iteration
        self.last = first_iteration + iterations - 1
        self.deadline = start_seconds + seconds
        self.byte_limit = byte_limit
        self.bytes_written = 0
        self.records = 0
        self.stop_reason = None

    def consume(self, row, now_seconds):
        if self.stop_reason is not None:
            return None
        if not math.isfinite(now_seconds):
            raise ValueError('invalid clock')
        if now_seconds >= self.deadline:
            self.stop_reason = 'time_budget'
            return None
        data = encode_observation(row)
        if row['iteration'] < self.first:
            raise ValueError('observation precedes complete-iteration boundary')
        if row['iteration'] > self.last:
            self.stop_reason = 'iteration_budget'
            return None
        if self.bytes_written + len(data) > self.byte_limit:
            self.stop_reason = 'byte_budget'
            return None
        self.bytes_written += len(data)
        self.records += 1
        return data

    def summary(self):
        return {'first_iteration': self.first, 'last_iteration': self.last,
                'bytes_written': self.bytes_written, 'records': self.records,
                'stop_reason': self.stop_reason,
                'coverage_verified': False}
