"""Bounded, lossless local monitor transport and profile command mailbox."""
import fcntl
import math
import mmap
import os
from pathlib import Path
import struct


CAPACITY = 512
_MAGIC = b'FLASHM01'
_HEADER = struct.Struct('<8s6Q')
_ROW = struct.Struct('<QdQd')
_SIZE = _HEADER.size + CAPACITY * _ROW.size


class MonitorChannel:
    @classmethod
    def create(cls, path):
        path = Path(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'r+b') as stream:
            stream.truncate(_SIZE)
            stream.write(_HEADER.pack(_MAGIC, 0, 0, 0, 0, 0, 0))
        return cls(path)

    def __init__(self, path):
        self.file = open(path, 'r+b', buffering=0)
        if os.fstat(self.file.fileno()).st_size != _SIZE:
            self.file.close()
            raise ValueError('invalid monitor channel size')
        self.memory = mmap.mmap(self.file.fileno(), _SIZE)
        if self.memory[:8] != _MAGIC:
            self.close()
            raise ValueError('invalid monitor channel version')

    def _lock(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)

    def _unlock(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)

    def publish(self, iteration, wall_iteration_ms, profile_epoch, completed_walltime):
        if iteration < 1 or not math.isfinite(wall_iteration_ms) or wall_iteration_ms <= 0:
            raise ValueError('monitor iteration and finite positive duration required')
        if profile_epoch < 0 or not math.isfinite(completed_walltime):
            raise ValueError('invalid monitor epoch or completion time')
        self._lock()
        try:
            header = list(_HEADER.unpack_from(self.memory))
            written, consumed = header[1:3]
            if written - consumed >= CAPACITY:
                raise BufferError('monitor channel full; refusing to overwrite unread iterations')
            if written:
                previous = _ROW.unpack_from(self.memory, _HEADER.size + (written - 1) % CAPACITY * _ROW.size)[0]
                if iteration != previous + 1:
                    raise ValueError('monitor iterations must be consecutive')
            _ROW.pack_into(self.memory, _HEADER.size + written % CAPACITY * _ROW.size,
                           iteration, wall_iteration_ms, profile_epoch, completed_walltime)
            struct.pack_into('<Q', self.memory, 8, written + 1)
        finally:
            self._unlock()

    def read_since(self, cursor):
        self._lock()
        try:
            header = list(_HEADER.unpack_from(self.memory))
            written = header[1]
            if cursor < max(0, written - CAPACITY) or cursor > written:
                raise BufferError('monitor cursor outside retained sequence')
            rows = [_ROW.unpack_from(self.memory, _HEADER.size + index % CAPACITY * _ROW.size)
                    for index in range(cursor, written)]
            header[2] = max(header[2], written)
            _HEADER.pack_into(self.memory, 0, *header)
        finally:
            self._unlock()
        return [dict(iteration=row[0], wall_iteration_ms=row[1], profile_epoch=row[2],
                     completed_walltime=row[3]) for row in rows], written

    def get_request(self):
        self._lock()
        try:
            return _HEADER.unpack_from(self.memory)[3]
        finally:
            self._unlock()

    def set_request(self, epoch):
        self._lock()
        try:
            header = list(_HEADER.unpack_from(self.memory))
            if epoch <= header[3]:
                raise ValueError('profile request epochs must strictly increase')
            header[3] = epoch
            _HEADER.pack_into(self.memory, 0, *header)
        finally:
            self._unlock()

    def set_schedule(self, epoch, start, end):
        self._lock()
        try:
            header = list(_HEADER.unpack_from(self.memory))
            if epoch != header[3] or epoch <= header[4] or end - start != 7 or start < 1:
                raise ValueError('invalid eight-step profile schedule')
            header[4:7] = [epoch, start, end]
            _HEADER.pack_into(self.memory, 0, *header)
        finally:
            self._unlock()

    def get_schedule(self):
        self._lock()
        try:
            epoch, start, end = _HEADER.unpack_from(self.memory)[4:7]
        finally:
            self._unlock()
        return dict(epoch=epoch, start=start, end=end) if epoch else None

    def close(self):
        if getattr(self, 'memory', None) is not None:
            self.memory.close()
            self.memory = None
        self.file.close()
