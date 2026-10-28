"""Batch exploratory scores for frozen R11 development observations only.

This tool is intentionally not run during implementation. It never traverses
validation/held-out entries and only reads rank JSONL files under observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flash_v4.offline_localizer import InputError, _read_observations, analyze


FROZEN_MATRIX = Path("results/v4/preflight/gpu_delay_collection_frozen_20261027_r11.json")
ALLOWED_TOPOLOGIES = ("legacy", "node_local_pp")


def _development_sessions(matrix_path: Path):
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise InputError("could not read the frozen collection matrix") from None
    if (matrix.get("status") != "frozen_collection_plan_before_first_r11_gpu_session"
            or matrix.get("release") != "v4_delay_20261027_r11"):
        raise InputError("input is not the frozen R11 collection matrix")
    sessions = matrix.get("session_schedule", {}).get("sessions")
    if not isinstance(sessions, list):
        raise InputError("frozen matrix has no session schedule")
    # Select the development partition before reading any case observations.
    development = [row for row in sessions
                   if isinstance(row, dict) and row.get("phase") == "development"]
    selected = {topology: [] for topology in ALLOWED_TOPOLOGIES}
    for row in development:
        topology = row.get("topology")
        if topology not in selected:
            continue
        role = None
        if row.get("condition") == "healthy" and str(row.get("session_id", "")).endswith("_H1"):
            role = "reference_h1"
        elif row.get("condition") == "healthy" and str(row.get("session_id", "")).endswith("_H2"):
            role = "healthy_repeat_h2"
        elif row.get("condition") == "fault":
            role = "fault_condition"
        if role is not None:
            selected[topology].append({
                "schedule_order": row.get("order"),
                "session_id": row.get("session_id"),
                "role": role,
            })
    for topology, rows in selected.items():
        references = [row for row in rows if row["role"] == "reference_h1"]
        if len(references) != 1:
            raise InputError("development matrix requires one H1 reference per topology")
        if not any(row["role"] == "healthy_repeat_h2" for row in rows):
            raise InputError("development matrix requires an H2 condition per topology")
    return matrix["release"], selected


def _case_root(release: str, session_id: str) -> Path:
    return Path("results/v4/compat") / release / session_id


def _is_complete(root: Path) -> bool:
    try:
        _read_observations(root)
    except (InputError, OSError):
        return False
    return True


def _coverage_summary(branch_result: dict[str, Any]) -> dict[str, Any]:
    ranks = branch_result["condition_model_coverage_by_rank"].values()
    total_events = known_events = unknown_events = 0
    total_weight = known_weight = 0.0
    for row in ranks:
        total_events += row["event_count"]
        known_events += row["known_event_count"]
        unknown_events += row["unknown_event_count"]
        total_weight += row["inverse_probability_weighted_events"]
        known_weight += row["known_inverse_probability_weighted_events"]
    return {
        "events": total_events,
        "known_events": known_events,
        "unknown_events": unknown_events,
        "known_weighted_event_coverage": known_weight / total_weight if total_weight else 0.0,
    }


def _top_rows(branch_result: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    rows = [row for row in branch_result["rank_order"] if row["score_median"] is not None]
    return [{"rank": row["rank"], "score": row["score_median"]} for row in rows[:limit]]


def _summary_case(case_key: str, topology: str, role: str, result: dict[str, Any]):
    branches = result["branches"]
    combined = result["combined_total_ratio"]
    combined_rows = [row for row in combined["rank_order"] if row["score_median"] is not None]
    top_combined = [{
        "rank": row["rank"],
        "score": row["score_median"],
        "iterations_with_both_branches": row["iterations_with_both_branches"],
    } for row in combined_rows[:3]]
    gemm_top, kernel_top = _top_rows(branches["gemm"]), _top_rows(branches["kernel"])
    gemm_top3 = {row["rank"] for row in gemm_top}
    kernel_top3 = {row["rank"] for row in kernel_top}
    gemm_top1 = gemm_top[0]["rank"] if gemm_top else None
    kernel_top1 = kernel_top[0]["rank"] if kernel_top else None
    paired_iterations = combined["common_drift_by_iteration"]
    rank_iteration_pairs = sum(
        row["iterations_with_both_branches"] for row in combined["rank_order"]
    )
    return {
        "case_key": case_key,
        "topology": topology,
        "condition_role": role,
        "status": "scored_exploration_only",
        "top3_combined_total_ratio": top_combined,
        "top3_gemm": gemm_top,
        "top3_kernel": kernel_top,
        "branch_consistency": {
            "gemm_kernel_top1_rank_agreement": (
                gemm_top1 == kernel_top1
                if gemm_top1 is not None and kernel_top1 is not None else None
            ),
            "gemm_kernel_top3_overlap_count": len(gemm_top3 & kernel_top3),
        },
        "coverage": {
            "gemm": _coverage_summary(branches["gemm"]),
            "kernel": _coverage_summary(branches["kernel"]),
        },
        "iterations_with_both_branches": {
            "iteration_ids_with_any_rank_data": sorted(int(k) for k in paired_iterations),
            "iteration_count_with_any_rank_data": len(paired_iterations),
            "rank_iteration_pairs": rank_iteration_pairs,
        },
        "accuracy_calculated": False,
        "threshold_or_alarm_applied": False,
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_batch(matrix_path: Path, output_root: Path) -> dict[str, Any]:
    release, sessions_by_topology = _development_sessions(matrix_path)
    output_root = output_root.expanduser().resolve()
    forbidden = {"observations", "actuation", "private", "truth", "validation", "held_out"}
    if any(part.lower() in forbidden for part in output_root.parts):
        raise InputError("output must be independent of observation and private-data trees")
    compat_root = Path("results/v4/compat").resolve()
    if output_root == compat_root or compat_root in output_root.parents:
        raise InputError("output must be outside the collected compatibility data tree")
    output_root.mkdir(parents=True, exist_ok=False)

    all_cases = []
    for topology in ALLOWED_TOPOLOGIES:
        scheduled = sessions_by_topology[topology]
        reference = next(row for row in scheduled if row["role"] == "reference_h1")
        reference_root = _case_root(release, reference["session_id"])
        reference_complete = _is_complete(reference_root)
        conditions = [row for row in scheduled if row["role"] != "reference_h1"]
        for ordinal, condition in enumerate(conditions, 1):
            case_key = f"{topology}_development_{ordinal:02d}"
            condition_root = _case_root(release, condition["session_id"])
            condition_complete = _is_complete(condition_root)
            if not reference_complete or not condition_complete:
                item = {
                    "case_key": case_key,
                    "topology": topology,
                    "condition_role": condition["role"],
                    "status": "collection_incomplete",
                    "reference_status": "complete" if reference_complete else "collection_incomplete",
                    "condition_status": "complete" if condition_complete else "collection_incomplete",
                    "accuracy_calculated": False,
                }
                _write_json_exclusive(output_root / f"{case_key}.json", item)
                all_cases.append(item)
                continue
            try:
                result = analyze(reference_root, condition_root)
            except InputError:
                item = {
                    "case_key": case_key,
                    "topology": topology,
                    "condition_role": condition["role"],
                    "status": "analysis_unavailable_insufficient_observations",
                    "accuracy_calculated": False,
                }
                _write_json_exclusive(output_root / f"{case_key}.json", item)
                all_cases.append(item)
                continue

            # The analyzer records input roots for standalone reproducibility;
            # batch reports replace them with opaque role labels so schedule
            # fault parameters do not leak into score artifacts.
            result.pop("inputs", None)
            result["batch_case"] = {
                "case_key": case_key,
                "topology": topology,
                "reference_role": "development_H1",
                "condition_role": condition["role"],
            }
            _write_json_exclusive(output_root / f"{case_key}.json", result)
            item = _summary_case(case_key, topology, condition["role"], result)
            all_cases.append(item)

    summary = {
        "schema": "flash_v4_development_batch_exploration_v1",
        "analysis_type": "development_rank_ordering_only",
        "validation_cases_read": False,
        "private_or_truth_data_read": False,
        "accuracy_calculated": False,
        "threshold_or_alarm_applied": False,
        "case_count": len(all_cases),
        "cases": all_cases,
    }
    _write_json_exclusive(output_root / "summary.json", summary)
    return summary


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=FROZEN_MATRIX)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory outside collected input trees")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_batch(args.matrix, args.output)
    except (InputError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(args.output),
        "case_count": summary["case_count"],
        "collection_incomplete_count": sum(
            row["status"] == "collection_incomplete" for row in summary["cases"]
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
