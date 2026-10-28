"""Bounded offline comparison of the pilot's GEMM observation envelope.

Only files matching the two supplied input roots' node*/observations/rank_*.jsonl
patterns are read. The output is always a new file and is never overwritten.

Sequence numbers are deliberately not part of operator identity. The output is
an offline compute-envelope comparison, not a pure kernel-speed measurement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any


TARGET_KINDS = frozenset(("gemm", "lt_matmul"))
MIN_SAMPLES_PER_DESCRIPTOR = 5
EXPECTED_RANKS = tuple(range(16))
RANK_FILE_RE = re.compile(r"rank_(\d+)\.jsonl$")


def _parse_iterations(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in raw.split(",")]
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("use comma-separated positive iteration numbers")
    try:
        iterations = [int(item) for item in values]
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated positive iteration numbers") from None
    if any(item < 1 for item in iterations) or len(set(iterations)) != len(iterations):
        raise argparse.ArgumentTypeError("iterations must be positive and unique")
    return tuple(sorted(iterations))


def _descriptor_fields(record: dict[str, Any]) -> tuple[int, str, Any, Any, tuple[int, ...]] | None:
    rank = record.get("rank")
    kind = record.get("kind")
    op = record.get("op")
    dtype = record.get("dtype")
    shape = record.get("shape")
    if isinstance(rank, bool) or not isinstance(rank, int):
        return None
    if not isinstance(kind, str) or kind not in TARGET_KINDS:
        return None
    if (isinstance(op, bool) or not isinstance(op, (str, int, float))
            or isinstance(dtype, bool) or not isinstance(dtype, (str, int, float))):
        return None
    for value in (op, dtype):
        if isinstance(value, float) and not math.isfinite(value):
            return None
    if not isinstance(shape, list) or not shape:
        return None
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape):
        return None
    return rank, kind, op, dtype, tuple(shape)


def _valid_elapsed(record: dict[str, Any]) -> float | None:
    value = record.get("elapsed_us")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    elapsed = float(value)
    if not math.isfinite(elapsed) or elapsed <= 0:
        return None
    return elapsed


def _read_group(
    group_root: Path,
    selected_iterations: tuple[int, ...] | None = None,
) -> tuple[
    dict[int, dict[tuple[Any, ...], list[float]]],
    dict[str, int],
    dict[int, dict[int, dict[tuple[Any, ...], int]]],
]:
    if group_root.is_symlink():
        raise ValueError("input group directory must not be a symlink")
    pattern = str(group_root / "node*" / "observations" / "rank_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no rank observation files matched {pattern}")

    by_rank: dict[int, dict[tuple[Any, ...], list[float]]] = {
        rank: defaultdict(list) for rank in EXPECTED_RANKS
    }
    iteration_counts: dict[int, dict[int, dict[tuple[Any, ...], int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    file_ranks: set[int] = set()
    counts = {
        "files": len(files),
        "target_kind_records": 0,
        "valid_target_records": 0,
        "selected_target_records": 0,
        "invalid_target_records": 0,
        "unassigned_iteration_records": 0,
        "filtered_out_target_records": 0,
    }
    for filename in files:
        source_path = Path(filename)
        if any(part.is_symlink() for part in (
            source_path, source_path.parent, source_path.parent.parent,
            source_path.parent.parent.parent,
        )):
            raise ValueError("observation inputs must be regular non-symlink files")
        match = RANK_FILE_RE.search(filename)
        if match is None:
            raise ValueError("observation filename did not contain a rank id")
        file_rank = int(match.group(1))
        if file_rank not in by_rank or file_rank in file_ranks:
            raise ValueError("expected one observation file for each global rank 0..15")
        file_ranks.add(file_rank)
        with open(filename, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("kind") not in TARGET_KINDS:
                    continue
                counts["target_kind_records"] += 1
                fields = _descriptor_fields(record)
                elapsed = _valid_elapsed(record)
                if fields is None or elapsed is None:
                    counts["invalid_target_records"] += 1
                    continue
                rank, kind, op, dtype, shape = fields
                if rank != file_rank:
                    raise ValueError(f"rank mismatch in {Path(filename).name}:{line_number}")
                descriptor = (kind, op, dtype, shape)
                counts["valid_target_records"] += 1
                if selected_iterations is not None:
                    iteration = record.get("iteration")
                    if (isinstance(iteration, bool) or not isinstance(iteration, int)
                            or iteration < 1):
                        counts["unassigned_iteration_records"] += 1
                        continue
                    if iteration not in selected_iterations:
                        counts["filtered_out_target_records"] += 1
                        continue
                    iteration_counts[iteration][rank][descriptor] += 1
                by_rank[rank][descriptor].append(elapsed)
                counts["selected_target_records"] += 1
    if file_ranks != set(EXPECTED_RANKS):
        raise ValueError("expected exactly one observation file for each global rank 0..15")
    return by_rank, counts, iteration_counts


def _validate_complete_iterations(
    selected_iterations: tuple[int, ...],
    health: dict[int, dict[int, dict[tuple[Any, ...], int]]],
    condition: dict[int, dict[int, dict[tuple[Any, ...], int]]],
    health_counts: dict[str, int],
    condition_counts: dict[str, int],
) -> None:
    if (health_counts["unassigned_iteration_records"]
            or condition_counts["unassigned_iteration_records"]):
        raise ValueError("selected iteration analysis requires iteration on every valid target record")
    for iteration in selected_iterations:
        health_ranks = health.get(iteration, {})
        condition_ranks = condition.get(iteration, {})
        if set(health_ranks) != set(EXPECTED_RANKS) or set(condition_ranks) != set(EXPECTED_RANKS):
            raise ValueError("selected iterations must be present on all 16 ranks in both inputs")
        for rank in EXPECTED_RANKS:
            if not health_ranks[rank] or health_ranks[rank] != condition_ranks[rank]:
                raise ValueError(
                    "selected iterations require matching descriptor/sample counts per rank"
                )


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _rank_comparison(
    rank: int,
    health: dict[tuple[Any, ...], list[float]],
    condition: dict[tuple[Any, ...], list[float]],
) -> dict[str, Any]:
    eligible = []
    ratio_values = []
    weighted_ratio_numerator = 0.0
    weighted_ratio_denominator = 0.0
    for descriptor in sorted(set(health) & set(condition), key=repr):
        health_values = health[descriptor]
        condition_values = condition[descriptor]
        if (len(health_values) < MIN_SAMPLES_PER_DESCRIPTOR
                or len(condition_values) < MIN_SAMPLES_PER_DESCRIPTOR):
            continue
        health_median = median(health_values)
        condition_median = median(condition_values)
        ratio = condition_median / health_median
        ratio_values.append(ratio)
        health_sample_elapsed_weight = sum(health_values)
        weighted_ratio_numerator += ratio * health_sample_elapsed_weight
        weighted_ratio_denominator += health_sample_elapsed_weight
        kind, op, dtype, shape = descriptor
        eligible.append({
            "descriptor": {
                "kind": kind,
                "op": op,
                "dtype": dtype,
                "shape": list(shape),
            },
            "health_samples": len(health_values),
            "condition_samples": len(condition_values),
            "health_median_elapsed_us": _round(health_median),
            "condition_median_elapsed_us": _round(condition_median),
            "condition_to_health_ratio": _round(ratio),
            "health_sample_elapsed_weight_us": _round(health_sample_elapsed_weight),
        })

    used_health = sum(item["health_samples"] for item in eligible)
    used_condition = sum(item["condition_samples"] for item in eligible)
    pairable_samples = sum(min(item["health_samples"], item["condition_samples"])
                           for item in eligible)
    total_health = sum(map(len, health.values()))
    total_condition = sum(map(len, condition.values()))
    health_coverage = used_health / total_health if total_health else None
    condition_coverage = used_condition / total_condition if total_condition else None
    median_ratio = median(ratio_values) if ratio_values else None
    weighted_ratio = (
        weighted_ratio_numerator / weighted_ratio_denominator
        if weighted_ratio_denominator else None
    )
    conservative_coverage = (
        min(health_coverage, condition_coverage)
        if health_coverage is not None and condition_coverage is not None else None
    )

    return {
        "rank": rank,
        "median_descriptor_time_ratio": (
            _round(median_ratio) if median_ratio is not None else None
        ),
        "median_descriptor_change_percent": (
            _round((median_ratio - 1) * 100, 3) if median_ratio is not None else None
        ),
        "health_sample_elapsed_weighted_ratio": (
            _round(weighted_ratio) if weighted_ratio is not None else None
        ),
        "health_sample_elapsed_weighted_change_percent": (
            _round((weighted_ratio - 1) * 100, 3) if weighted_ratio is not None else None
        ),
        "health_sample_elapsed_weight_total_us": _round(weighted_ratio_denominator),
        "common_descriptor_count": len(eligible),
        "valid_health_sample_count": total_health,
        "valid_condition_sample_count": total_condition,
        "paired_sample_coverage": {
            "health_used_samples": used_health,
            "health_total_valid_samples": total_health,
            "health_percent": (
                _round(health_coverage * 100, 3) if health_coverage is not None else None
            ),
            "condition_used_samples": used_condition,
            "condition_total_valid_samples": total_condition,
            "condition_percent": (
                _round(condition_coverage * 100, 3)
                if condition_coverage is not None else None
            ),
            "pairable_sample_count": pairable_samples,
            "conservative_min_side_percent": (
                _round(conservative_coverage * 100, 3)
                if conservative_coverage is not None else None
            ),
        },
        "descriptors": eligible,
    }


def analyze(
    healthy_root: Path,
    fault_root: Path,
    selected_iterations: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if healthy_root.resolve() == fault_root.resolve():
        raise ValueError("healthy and comparison input directories must differ")
    health_data, health_counts, health_iterations = _read_group(
        healthy_root, selected_iterations)
    condition_data, condition_counts, condition_iterations = _read_group(
        fault_root, selected_iterations)
    if selected_iterations is not None:
        _validate_complete_iterations(
            selected_iterations, health_iterations, condition_iterations,
            health_counts, condition_counts,
        )
    rank_rows = [
        _rank_comparison(rank, health_data[rank], condition_data[rank])
        for rank in EXPECTED_RANKS
    ]
    changed = [row for row in rank_rows if row["median_descriptor_time_ratio"] is not None]
    top_changes = sorted(
        changed,
        key=lambda row: (-abs(row["median_descriptor_time_ratio"] - 1), row["rank"]),
    )[:3]
    return {
        "analysis": "bounded_offline_compute_envelope_comparison",
        "status": "unfrozen_exploration",
        "inputs": {"health_reference": str(healthy_root), "condition_comparison": str(fault_root)},
        "iteration_selection": list(selected_iterations) if selected_iterations is not None else "all",
        "conclusion_limits": [
            "does not verify effective fault application",
            "does not establish a pass/fail result",
            "does not freeze an algorithm",
        ],
        "method": {
            "target_kinds": sorted(TARGET_KINDS),
            "descriptor_identity": ["rank", "kind", "op", "dtype", "shape"],
            "sequence_used_as_identity": False,
            "validity": "finite positive elapsed_us and scalar op/dtype with positive integer shape",
            "minimum_samples_per_group_descriptor": MIN_SAMPLES_PER_DESCRIPTOR,
            "descriptor_ratio": "condition descriptor median elapsed_us / health descriptor median elapsed_us",
            "rank_median": "unweighted median across eligible descriptor ratios",
            "health_sample_elapsed_weighted_ratio": (
                "weighted mean of descriptor ratios, using the sum of healthy elapsed_us samples "
                "for each eligible descriptor as its weight; this is a sample-distribution weight, "
                "not total GPU time"
            ),
            "paired_sample_coverage": (
                "eligible-descriptor samples divided by all valid target-kind samples per side; "
                "conservative_min_side_percent is the lower side percentage"
            ),
            "selected_iteration_completeness": (
                "each requested iteration must appear on all ranks in both inputs with matching "
                "per-rank descriptor sets and sample counts"
            ),
            "top_three_order": "descending absolute deviation of rank median ratio from 1",
            "interpretation": "compute-envelope change, not pure kernel speed",
        },
        "input_counts": {
            "health_reference": health_counts,
            "condition_comparison": condition_counts,
        },
        "ranks": rank_rows,
        "top_three_changes": [
            {
                "rank": row["rank"],
                "median_descriptor_time_ratio": row["median_descriptor_time_ratio"],
                "median_descriptor_change_percent": row["median_descriptor_change_percent"],
                "health_sample_elapsed_weighted_ratio": row[
                    "health_sample_elapsed_weighted_ratio"
                ],
                "health_sample_elapsed_weighted_change_percent": row[
                    "health_sample_elapsed_weighted_change_percent"
                ],
                "common_descriptor_count": row["common_descriptor_count"],
                "conservative_min_side_coverage_percent": row["paired_sample_coverage"][
                    "conservative_min_side_percent"
                ],
            }
            for row in top_changes
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare GEMM compute envelopes from two public observation directories."
    )
    parser.add_argument(
        "--healthy", required=True, type=Path,
        help="health reference directory containing node*/observations/rank_*.jsonl",
    )
    parser.add_argument(
        "--fault", required=True, type=Path,
        help="comparison directory containing node*/observations/rank_*.jsonl",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="new JSON report path; existing files are never overwritten",
    )
    parser.add_argument(
        "--iterations", type=_parse_iterations, default=None,
        help="optional comma-separated complete public iterations; default uses all iterations",
    )
    return parser


def _output_is_within_input(output: Path, input_root: Path) -> bool:
    resolved_output = output.resolve()
    resolved_input = input_root.resolve()
    return resolved_output == resolved_input or resolved_input in resolved_output.parents


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.healthy.resolve() == args.fault.resolve():
        parser.error("--healthy and --fault must identify different directories")
    if (_output_is_within_input(args.output, args.healthy)
            or _output_is_within_input(args.output, args.fault)):
        parser.error("--output must be outside both input directories")
    if args.output.exists():
        parser.error("--output already exists; refusing to overwrite it")

    try:
        result = analyze(args.healthy, args.fault, args.iterations)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        parser.error("--output was created concurrently; refusing to overwrite it")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
