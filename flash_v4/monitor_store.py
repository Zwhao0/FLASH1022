"""Opt-in per-iteration Store notification; removes control rendezvous by explicit user authorization."""
from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
import math


class ControlTimeout(TimeoutError):
    """A peer did not observe a root command before its configured deadline."""


class StoreControl:
    """Publish/read one immutable command per control iteration.

    ``store`` is expected to provide PyTorch-Store-like ``set(key, bytes)``,
    ``wait([key], timedelta)``, and ``get(key)`` methods. The caller must supply
    one unique run_id shared by all ranks; its hash namespaces keys without
    importing torch here. Keys remain in the backing store until that store is
    cleaned up because this candidate deliberately has no peer acknowledgments.
    """

    ROOT = 0
    COMMANDS = frozenset((-1, 0, 1))

    def __init__(self, store, *, run_id, rank, world_size=16,
                 wait_timeout_seconds=30.0):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if type(rank) is not int or type(world_size) is not int:
            raise TypeError("rank and world_size must be integers")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be inside world_size")
        if (isinstance(wait_timeout_seconds, bool)
                or not isinstance(wait_timeout_seconds, (int, float))
                or not math.isfinite(wait_timeout_seconds)
                or wait_timeout_seconds <= 0):
            raise ValueError("wait_timeout_seconds must be finite and positive")

        self.store = store
        self.rank = rank
        self.world_size = world_size
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        run_key = sha256(run_id.encode("utf-8")).hexdigest()
        self.prefix = f"flash_v4_monitor/r92/{run_key}"
        self._last_iteration = -1

    def _key(self, iteration):
        return f"{self.prefix}/iteration/{iteration}/command"

    def _validate_iteration(self, iteration):
        if type(iteration) is not int:
            raise TypeError("iteration must be an integer")
        if iteration != 0 and (iteration < 4 or (iteration - 4) % 8 != 0):
            raise ValueError("iteration must be startup 0 or a 4 mod 8 boundary")
        if iteration <= self._last_iteration:
            raise ValueError("duplicate or out-of-order iteration")

    def publish(self, iteration, command):
        """Root publishes one immutable command; this operation has no ACK."""
        if self.rank != self.ROOT:
            raise PermissionError("only rank zero may publish commands")
        self._validate_iteration(iteration)
        if type(command) is not int or command not in self.COMMANDS:
            raise ValueError("command must be exactly -1, 0, or 1")
        self.store.set(self._key(iteration), str(command).encode("ascii"))
        self._last_iteration = iteration

    def receive(self, iteration):
        """A peer waits for and reads the command for this exact iteration."""
        if self.rank == self.ROOT:
            raise PermissionError("rank zero publishes rather than receives")
        self._validate_iteration(iteration)
        key = self._key(iteration)
        try:
            ready = self.store.wait(
                [key], timedelta(seconds=self.wait_timeout_seconds))
        except TimeoutError as error:
            raise ControlTimeout(
                f"timed out waiting for command at iteration {iteration}") from error
        if ready is False:
            raise ControlTimeout(
                f"timed out waiting for command at iteration {iteration}")

        raw = self.store.get(key)
        try:
            text = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
            command = int(text)
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            raise ValueError("invalid stored command") from error
        if text not in ("-1", "0", "1") or command not in self.COMMANDS:
            raise ValueError("stored command must be exactly -1, 0, or 1")
        self._last_iteration = iteration
        return command


CONTRACT = 'r92_store_v1'

def create_control(dist, run_id, rank):
    import re
    if not isinstance(run_id,str) or not re.fullmatch(r'v4_[a-f0-9]{32}',run_id):
        raise ValueError('Store control requires a unique supervised run identity')
    if dist.get_world_size()!=16:
        raise ValueError('Store runtime requires the explicit sixteen-rank contract')
    return StoreControl(dist.distributed_c10d._get_default_store(),run_id=run_id,rank=rank,world_size=16)
