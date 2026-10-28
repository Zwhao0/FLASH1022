"""Run the current v1 trainer with bounded v4 runtime adaptations.

This module intentionally uses only the Python standard library until the
per-rank MPS environment has been consumed and set.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, MutableMapping, Sequence


PRIVATE_QUOTA_ENV = "FLASH_V4_QUOTA_PRIVATE"
PRIVATE_ACTUATION_DIR_ENV = "FLASH_V4_ACTUATION_DIR_PRIVATE"
PRIVATE_STREAM_DELAY_ENV = "FLASH_V4_STREAM_DELAY_PRIVATE"
PRIVATE_ACTOR_WINDOW_ENV = "FLASH_V4_ACTOR_WINDOW_PRIVATE"
MPS_QUOTA_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
LAYOUT_ENV = "FLASH_PARALLEL_LAYOUT"
RUNTIME_METADATA = "runtime_adaptation.json"
WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
PYTORCH_ALLOCATOR_FRACTION = 0.03
_LAYOUTS = ("legacy", "node_local_pp")
_ACTIVE_OBSERVER = None
_ACTIVE_STREAM_DELAY = None
_DISABLED_FLAGS = {
    "--collection-plan",
    "--online",
    "--monitor-benchmark",
    "--compute-delay-ms",
    "--fault-rank",
    "--delay-start",
    "--delay-end",
    "--delay-placement",
}


@dataclass(frozen=True)
class ProcessStartup:
    rank: int
    local_rank: int
    requested_quota_percent: int
    actuation_dir: Path | None
    stream_delay_enabled: bool = False
    stream_delay_ratio: float = 0.0
    stream_delay_first: int = 4
    stream_delay_last: int = 11


class _ClosedResource:
    """Tiny close-compatible stand-in used by the no-op Redis writer."""

    def close(self) -> None:
        pass


class LocalNullRedisWriter:
    """Keep the v1 writer interface without opening a Redis connection."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.reader = _ClosedResource()
        self.connection = _ClosedResource()

    def write(self, iteration: int, milliseconds: float) -> None:
        pass

    def close(self) -> None:
        self.reader.close()
        self.connection.close()


class NoOpGPUStreamDelay:
    """Health-run delay object; it deliberately installs no module hooks."""

    def __init__(self, args: object, device: object, output: object) -> None:
        self.interval = None
        self.cycles = 0
        self.active = False

    def attach_backward(self, model: object) -> None:
        pass

    def begin_iteration(self, iteration: int) -> None:
        pass

    def apply(self, iteration: int) -> None:
        pass

    def set_profile_enabled(self, enabled: bool) -> None:
        pass

    def close(self, iteration: int) -> None:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v1 PP2/EP2 trainer through the v4 runtime adapter",
        allow_abbrev=False,
    )
    parser.add_argument("--layout", required=True, choices=_LAYOUTS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default="models/qwen_moe")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    return parser


def _reject_disabled_flags(argv: Sequence[str]) -> None:
    for token in argv:
        option = token.split("=", 1)[0]
        if option in _DISABLED_FLAGS:
            raise ValueError(f"{option} is disabled by the v4 health entry")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    _reject_disabled_flags(values)
    args = _build_parser().parse_args(values)
    if args.steps < 1 or args.warmup < 0:
        raise ValueError("steps must be positive and warmup nonnegative")
    if args.seq_len < 1 or args.batch_size < 1 or args.microbatches < 1:
        raise ValueError("sequence length, batch size, and microbatches must be positive")
    return args


def _assert_cuda_not_initialized() -> None:
    """Check only an already-imported torch module; never import torch here."""
    torch_module = sys.modules.get("torch")
    cuda_module = getattr(torch_module, "cuda", None) if torch_module is not None else None
    is_initialized = getattr(cuda_module, "is_initialized", None)
    if callable(is_initialized) and is_initialized():
        raise RuntimeError("MPS quota must be configured before CUDA initialization")


def _validate_launcher_environment(environ: Mapping[str, str]) -> tuple[int, int]:
    def read_int(name: str) -> int:
        raw = environ.get(name)
        try:
            return int(raw) if raw is not None else -1
        except (TypeError, ValueError):
            return -1

    world_size = read_int("WORLD_SIZE")
    local_world_size = read_int("LOCAL_WORLD_SIZE")
    rank = read_int("RANK")
    local_rank = read_int("LOCAL_RANK")
    if world_size != WORLD_SIZE or local_world_size != LOCAL_WORLD_SIZE:
        raise ValueError("v4 training requires WORLD_SIZE=16 and LOCAL_WORLD_SIZE=8")
    if not 0 <= rank < WORLD_SIZE or not 0 <= local_rank < LOCAL_WORLD_SIZE:
        raise ValueError("v4 training requires valid torchrun RANK and LOCAL_RANK")
    return rank, local_rank


def _validate_preload(environ: Mapping[str, str]) -> None:
    preload = environ.get("LD_PRELOAD", "")
    if any("flashprobe" in part.lower() for part in preload.split(os.pathsep) if part):
        raise ValueError("legacy FLASH LD_PRELOAD probe detected; unset it before torchrun")


def _parse_quota_mapping(raw: str | None) -> dict[int, int]:
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("FLASH_V4_QUOTA_PRIVATE must be a JSON rank-to-quota mapping") from None
    if not isinstance(parsed, dict):
        raise ValueError("FLASH_V4_QUOTA_PRIVATE must be a JSON rank-to-quota mapping")

    result: dict[int, int] = {}
    for key, value in parsed.items():
        try:
            rank = int(key)
        except (TypeError, ValueError):
            raise ValueError("FLASH_V4_QUOTA_PRIVATE contains an invalid rank") from None
        if str(rank) != str(key) or not 0 <= rank < WORLD_SIZE:
            raise ValueError("FLASH_V4_QUOTA_PRIVATE contains an invalid rank")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("FLASH_V4_QUOTA_PRIVATE contains an invalid quota")
        numeric = float(value)
        if not math.isfinite(numeric) or not 1 <= numeric <= 100 or not numeric.is_integer():
            raise ValueError("FLASH_V4_QUOTA_PRIVATE contains an invalid quota")
        result[rank] = int(numeric)
    return result


def _parse_stream_delay_mapping(raw: str | None) -> dict[int, float]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError(f"{PRIVATE_STREAM_DELAY_ENV} must be a JSON rank-to-ratio mapping") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{PRIVATE_STREAM_DELAY_ENV} must be a JSON rank-to-ratio mapping")

    result: dict[int, float] = {}
    for key, value in parsed.items():
        try:
            rank = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"{PRIVATE_STREAM_DELAY_ENV} contains an invalid rank") from None
        if str(rank) != str(key) or not 0 <= rank < WORLD_SIZE:
            raise ValueError(f"{PRIVATE_STREAM_DELAY_ENV} contains an invalid rank")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{PRIVATE_STREAM_DELAY_ENV} contains an invalid ratio")
        ratio = float(value)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 8.0:
            raise ValueError(f"{PRIVATE_STREAM_DELAY_ENV} contains an invalid ratio")
        result[rank] = ratio
    return result


def _parse_actor_window(raw: str | None) -> tuple[int, int]:
    if raw is None:
        return 4, 11
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError(f"{PRIVATE_ACTOR_WINDOW_ENV} must be a JSON first/last window") from None
    if not isinstance(parsed, dict) or set(parsed) != {"first", "last"}:
        raise ValueError(f"{PRIVATE_ACTOR_WINDOW_ENV} must be a JSON first/last window")
    first, last = parsed["first"], parsed["last"]
    if type(first) is not int or type(last) is not int:
        raise ValueError(f"{PRIVATE_ACTOR_WINDOW_ENV} requires integer first and last")
    if not (4 <= first <= last <= 120) or last - first + 1 > 32:
        raise ValueError(f"{PRIVATE_ACTOR_WINDOW_ENV} is outside the allowed bounded window")
    return first, last


def _parse_private_actuation_dir(
    raw: str | None,
    training_output: str | Path | None,
) -> Path | None:
    if raw is None or raw == "":
        return None
    try:
        actuation_dir = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError):
        raise ValueError("FLASH_V4_ACTUATION_DIR_PRIVATE is not a usable path") from None
    if training_output is not None:
        output_dir = Path(training_output).expanduser().resolve()
        if (actuation_dir == output_dir or actuation_dir in output_dir.parents
                or output_dir in actuation_dir.parents):
            raise ValueError("private actuation evidence must be separate from training output")
    return actuation_dir


def _configure_process_environment(
    layout: str,
    environ: MutableMapping[str, str] | None = None,
    training_output: str | Path | None = None,
) -> ProcessStartup:
    env = os.environ if environ is None else environ
    # Consume the private controller payload before any framework import or
    # validation that might raise, and never include its contents in errors.
    private_actuation_dir = env.pop(PRIVATE_ACTUATION_DIR_ENV, None)
    private_quota = env.pop(PRIVATE_QUOTA_ENV, None)
    private_stream_delay = env.pop(PRIVATE_STREAM_DELAY_ENV, None)
    private_actor_window = env.pop(PRIVATE_ACTOR_WINDOW_ENV, None)
    _assert_cuda_not_initialized()
    if layout not in _LAYOUTS:
        raise ValueError("layout must be legacy or node_local_pp")
    rank, local_rank = _validate_launcher_environment(env)
    _validate_preload(env)
    actuation_dir = _parse_private_actuation_dir(private_actuation_dir, training_output)
    quotas = _parse_quota_mapping(private_quota)
    stream_delay_mapping = _parse_stream_delay_mapping(private_stream_delay)
    stream_delay_first, stream_delay_last = _parse_actor_window(private_actor_window)
    quota = quotas.get(rank, 100)
    stream_delay_enabled = private_stream_delay is not None
    if private_actor_window is not None and not stream_delay_enabled:
        raise ValueError("actor window requires a stream-delay actor mapping")
    if stream_delay_enabled and quota != 100:
        raise ValueError("stream-delay actor requires CUDA MPS quota 100 on every rank")
    stream_delay_ratio = stream_delay_mapping.get(rank, 0.0)
    if stream_delay_enabled and actuation_dir is None:
        raise ValueError("stream-delay actor requires a private actuation directory")
    del (quotas, private_quota, private_actuation_dir, private_stream_delay,
         private_actor_window, stream_delay_mapping)
    env[MPS_QUOTA_ENV] = str(quota)
    env[LAYOUT_ENV] = layout
    return ProcessStartup(
        rank, local_rank, quota, actuation_dir,
        stream_delay_enabled, stream_delay_ratio, stream_delay_first, stream_delay_last,
    )


def _v1_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--output", str(Path(args.output).resolve()),
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
        "--seq-len", str(args.seq_len),
        "--seed", str(args.seed),
        "--model", str(args.model),
        "--batch-size", str(args.batch_size),
        "--microbatches", str(args.microbatches),
        "--tensor-parallel-size", "2",
        "--expert-parallel-size", "2",
    ]


def _write_runtime_metadata(
    output: str | Path,
    layout: str,
    stream_delay_actor_enabled: bool = False,
) -> None:
    path = Path(output).resolve()
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "entry": "flash_v4.train_entry",
        "layout": layout,
        "runtime_adaptations": [
            "per_rank_mps_startup_environment",
            "local_null_redis_writer",
            "no_op_gpu_stream_delay",
            "legacy_probe_disabled",
        ],
        "shared_compatibility_resource_budget": {
            "pytorch_allocator_fraction": PYTORCH_ALLOCATOR_FRACTION,
            "scope": "this_process_only",
        },
    }
    if stream_delay_actor_enabled:
        metadata["runtime_adaptations"].append("native_stream_delay_schedule_actor")
    (path / RUNTIME_METADATA).write_text(json.dumps(metadata, indent=2) + "\n")


def _patch_v1_training_module(training_module: object, startup: ProcessStartup) -> None:
    setattr(training_module, "RedisWriter", LocalNullRedisWriter)
    setattr(training_module, "GPUStreamDelay", NoOpGPUStreamDelay)
    torch_module = getattr(training_module, "torch", None)
    if torch_module is None:
        raise RuntimeError("v1 training module did not expose torch for allocator budgeting")
    _install_allocator_budget(torch_module, startup)


def _local_device_index(device: object, fallback: int) -> int:
    if isinstance(device, int):
        return device
    index = getattr(device, "index", None)
    return fallback if index is None else int(index)


def _write_actuation_evidence(
    startup: ProcessStartup,
    torch_module: object,
    device: object,
    allocator_budget: dict | None = None,
) -> None:
    if startup.actuation_dir is None:
        return
    cuda_module = getattr(torch_module, "cuda", None)
    get_properties = getattr(cuda_module, "get_device_properties", None)
    if not callable(get_properties):
        raise RuntimeError("PyTorch device-properties API is unavailable for private evidence")
    try:
        properties = get_properties(device)
        sm_count = int(getattr(properties, "multi_processor_count"))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        raise RuntimeError("could not read the selected CUDA device properties") from None
    if sm_count < 1:
        raise RuntimeError("selected CUDA device reported an invalid multiprocessor count")

    device_uuid = None
    for name in ("uuid", "device_uuid"):
        value = getattr(properties, name, None)
        if value is not None and str(value):
            device_uuid = str(value)
            break
    is_initialized = getattr(cuda_module, "is_initialized", None)
    context_initialized = bool(is_initialized()) if callable(is_initialized) else None
    evidence = {
        "schema": "flash_v4_private_actuation_evidence_v1",
        "rank": startup.rank,
        "local_rank": startup.local_rank,
        "local_device": _local_device_index(device, startup.local_rank),
        "device_uuid": device_uuid,
        "multi_processor_count": sm_count,
        "requested_mps_quota_percent": startup.requested_quota_percent,
        "effective_mps_quota_percent": None,
        "effective_quota_verified": False,
        "effective_quota_status": "not exposed by PyTorch CUDA device properties",
        "request_set_before_device_selection": True,
        "cuda_context_initialized": context_initialized,
        "pytorch_allocator_fraction": (allocator_budget or {}).get("fraction", PYTORCH_ALLOCATOR_FRACTION),
        "allocator_budget": allocator_budget,
        "captured_at_unix": time.time(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    evidence_dir = startup.actuation_dir
    try:
        evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        evidence_file = evidence_dir / f"rank_{startup.rank}.json"
        fd = os.open(evidence_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise RuntimeError("could not create unique private actuation evidence for this rank") from None


def _install_allocator_budget(torch_module: object, startup: ProcessStartup) -> None:
    """Apply a process-local allocator cap immediately after v1 selects a device."""
    cuda_module = getattr(torch_module, "cuda", None)
    set_device = getattr(cuda_module, "set_device", None)
    set_fraction = getattr(cuda_module, "set_per_process_memory_fraction", None)
    if not callable(set_device) or not callable(set_fraction):
        raise RuntimeError("PyTorch CUDA allocator budget API is unavailable")
    if getattr(set_device, "_flash_v4_allocator_budget", False):
        return

    evidence_written = False
    allocator_budget = None
    budget_mode = os.environ.get("FLASH_V4_ALLOCATOR_BUDGET", "fixed")
    if budget_mode not in ("fixed", "available"):
        raise ValueError("allocator budget must be fixed or available")

    def set_device_with_budget(device: object = None) -> object:
        nonlocal evidence_written, allocator_budget
        result = set_device(device)
        if allocator_budget is None:
            if budget_mode == "available":
                free_bytes, total_bytes = cuda_module.mem_get_info(device=device)
                if not 0 < free_bytes <= total_bytes:
                    raise RuntimeError("selected GPU reported invalid available memory")
                allocator_budget = dict(mode=budget_mode, free_bytes_at_setup=int(free_bytes),
                                        total_bytes=int(total_bytes), limit_bytes=int(free_bytes),
                                        fraction=free_bytes / total_bytes)
            else:
                allocator_budget = dict(mode=budget_mode, fraction=PYTORCH_ALLOCATOR_FRACTION)
        set_fraction(allocator_budget["fraction"], device=device)
        if not evidence_written:
            _write_actuation_evidence(startup, torch_module, device, allocator_budget)
            evidence_written = True
        return result

    setattr(set_device_with_budget, "_flash_v4_allocator_budget", True)
    # The opt-in finalizer must select this thread's device without reapplying
    # the process allocator budget while the optimizer may be running.
    setattr(set_device_with_budget, "_flash_v4_device_only", set_device)
    setattr(cuda_module, "set_device", set_device_with_budget)


def _load_v1_train_pipeline(startup: ProcessStartup):
    global _ACTIVE_OBSERVER
    global _ACTIVE_STREAM_DELAY
    if "flash_v1.train_pipeline" in sys.modules:
        raise RuntimeError("v1 train_pipeline was imported before v4 runtime patching")
    training_module = importlib.import_module("flash_v1.train")
    _patch_v1_training_module(training_module, startup)
    from flash_v4.observer_runtime import install
    _ACTIVE_OBSERVER = install(training_module, NoOpGPUStreamDelay)
    from flash_v4.stream_delay_runtime import install as install_stream_delay
    _ACTIVE_STREAM_DELAY = install_stream_delay(training_module, startup)
    return importlib.import_module("flash_v1.train_pipeline")


def _best_effort_cuda_synchronize() -> None:
    """Drain queued device work on both normal and exceptional exits."""
    torch_module = sys.modules.get("torch")
    cuda_module = getattr(torch_module, "cuda", None) if torch_module is not None else None
    is_initialized = getattr(cuda_module, "is_initialized", None)
    synchronize = getattr(cuda_module, "synchronize", None)
    if callable(is_initialized) and callable(synchronize):
        try:
            if is_initialized():
                synchronize()
        except BaseException:
            # Do not hide an exception already unwinding from the trainer.
            pass


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = _parse_args(argv)
        startup = _configure_process_environment(
            args.layout,
            training_output=args.output,
        )
        if startup.rank == 0:
            _write_runtime_metadata(args.output, args.layout, startup.stream_delay_enabled)

        v1_argv = _v1_argv(args)
        sys.argv = ["flash_v1.train_pipeline", *v1_argv]
        train_pipeline = _load_v1_train_pipeline(startup)
        train_pipeline.main()
    finally:
        _best_effort_cuda_synchronize()
        for runtime in (_ACTIVE_STREAM_DELAY, _ACTIVE_OBSERVER):
            if runtime is not None:
                try:
                    runtime.flush()
                except BaseException:
                    # Preserve the trainer's active exception on controlled exit.
                    pass


if __name__ == "__main__":
    main()
