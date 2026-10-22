"""Interpretable signed-wait localization for the fixed FLASH Qwen TP/PP scope.

This is an offline research extension. It uses paired first-call durations and
healthy-only limits; it is not the paper's original fingerprint algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash_v1.offline import (
    FIRST_PHASE_BLOCK_COUNT,
    FIRST_PHASE_BLOCK_SIZE,
    FIRST_PHASE_SAMPLING_STRIDE,
    FeatureSelectionUnavailable,
    _canonical_json,
    _contiguous_runs,
    _is_tensor_parallel,
    _profile_rows,
    _read_json,
    _row_phase_code,
    _sha256_bytes,
    _stable_groups,
    _validate_first_phase_config,
    extract_run_features_data,
)


MODEL_VERSION = 1
WORLD_SIZE = 16
PIPELINE_STAGES = 2
TENSOR_PARALLEL_SIZE = 2
TOTAL_LAYERS = 24
PHASES_PER_RANK = TOTAL_LAYERS // PIPELINE_STAGES
EXPECTED_RANKS = tuple(range(WORLD_SIZE))
HEALTHY_LIMIT_MULTIPLIER = 1.25
FAULT_LIMIT_MULTIPLIER = 2.0
MIN_TOP_RATIO = 2.0
SCORE_FLOOR_US = 1.0


class CoverageError(ValueError):
    """A run does not contain a complete, pairable attention-BWD signature."""


@dataclass(frozen=True)
class RunScores:
    run_id: str
    rank_scores_us: dict[int, float]
    phase_scores_us: dict[int, dict[int, float]]
    coverage: dict[str, Any]
    source_hashes: dict[str, str]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be an object: {run_dir / 'manifest.json'}")
    return manifest


def _validate_scope(manifest: dict[str, Any], topology: Any) -> None:
    try:
        _validate_first_phase_config(manifest)
    except FeatureSelectionUnavailable as error:
        raise CoverageError(str(error)) from error
    config = manifest.get("configuration", manifest.get("config"))
    expected = {
        "tp": TENSOR_PARALLEL_SIZE,
        "pp": PIPELINE_STAGES,
        "world_size": WORLD_SIZE,
        "microbatches": FIRST_PHASE_BLOCK_COUNT,
        "attention_backward_block_size": FIRST_PHASE_BLOCK_SIZE,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if type(actual) is not int or actual != value:
            raise CoverageError(f"wait_localize requires manifest {key}={value}; got {actual!r}")
    if not isinstance(topology, dict) or topology.get("world_size") != WORLD_SIZE:
        raise CoverageError(f"wait_localize requires topology world_size={WORLD_SIZE}")
    # This explicit architecture assumption is part of the frozen method spec:
    # the configured 24-layer model is split over two PP stages, yielding 12
    # attention-backward phase identities per rank.
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise CoverageError("wait_localize requires an explicit model_id in manifest configuration")


def _stable_tp_group(
    stable: dict[tuple[tuple[int, ...], str, str], dict[str, Any]],
    members: tuple[int, ...],
    tag: str,
    group_name: str,
) -> dict[str, Any] | None:
    group = stable.get((members, tag, group_name))
    if group is not None:
        return group
    candidates = [
        item for (candidate_members, label, name), item in stable.items()
        if candidate_members == members and name == group_name and tag.startswith(label + ":")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _extract_wait_scores(
    run_dir: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> RunScores:
    manifest = manifest or _read_run_manifest(run_dir)
    topology_path = run_dir / "topology.json"
    profile_path = run_dir / "profile.json"
    if not topology_path.is_file() or not profile_path.is_file():
        raise CoverageError(f"{run_dir.name}: expected topology.json and profile.json")
    topology = _read_json(topology_path)
    profile = _read_json(profile_path)
    _validate_scope(manifest, topology)

    # Reuse the first_in_phase selector as the strict evidence/coverage gate.
    # The raw-duration calculation below preserves exact (iteration,index) pairs.
    try:
        extract_run_features_data(
            topology, profile, phases="attention_backward", call_selection="first_in_phase"
        )
    except FeatureSelectionUnavailable as error:
        raise CoverageError(str(error)) from error

    stable = _stable_groups(topology)
    tp_groups = {
        (members, label, name): group
        for (members, label, name), group in stable.items()
        if _is_tensor_parallel(str(group.get("name", ""))) and len(members) == 2
    }
    if len(tp_groups) != WORLD_SIZE // TENSOR_PARALLEL_SIZE:
        raise CoverageError(
            f"expected {WORLD_SIZE // TENSOR_PARALLEL_SIZE} stable tensor_parallel groups; "
            f"found {len(tp_groups)}"
        )
    member_counts = {rank: 0 for rank in EXPECTED_RANKS}
    for members, _label, _name in tp_groups:
        for rank in members:
            if rank not in member_counts:
                raise CoverageError(f"unexpected tensor_parallel rank {rank}")
            member_counts[rank] += 1
    if any(count != 1 for count in member_counts.values()):
        raise CoverageError("topology must assign every rank to exactly one TP2 group")

    index_sets: dict[tuple[Any, ...], set[int]] = {}
    pair_rows: dict[tuple[Any, ...], dict[int, float]] = {}
    for row in _profile_rows(profile):
        group_name = row.get("group_name")
        if not isinstance(group_name, str) or not _is_tensor_parallel(group_name):
            continue
        members_value = row.get("group_members")
        if not isinstance(members_value, list) or any(type(rank) is not int for rank in members_value):
            raise CoverageError("tensor_parallel row lacks actual group_members")
        members = tuple(sorted(members_value))
        tag = row.get("tag")
        if not isinstance(tag, str):
            raise CoverageError("tensor_parallel row lacks semantic tag")
        group = _stable_tp_group(stable, members, tag, group_name)
        if group is None:
            continue
        phase_code = _row_phase_code(row)
        if phase_code is None:
            raise CoverageError("tensor_parallel row has invalid phase_code")
        if (phase_code & 0xFF) != 2:
            continue

        operation_index = row.get("operation_index")
        iteration = row.get("iteration")
        rank = row.get("rank")
        duration = row.get("duration_us")
        count = row.get("count")
        dtype = row.get("dtype")
        kind = row.get("kind", row.get("operation", group.get("kind", "unspecified")))
        if type(operation_index) is not int or operation_index < 0:
            raise CoverageError("attention_backward row lacks operation_index")
        if type(iteration) is not int or iteration < 0 or type(rank) is not int or rank not in members:
            raise CoverageError("attention_backward row lacks valid iteration/rank")
        if count is None or dtype is None:
            raise CoverageError("attention_backward row lacks count/dtype")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or
                not math.isfinite(float(duration)) or float(duration) <= 0):
            raise CoverageError("attention_backward row has invalid duration_us")

        label = str(group.get("label", group.get("semantic_label", group.get("tag", tag))))
        identity = (members, label, group_name, str(kind), str(count), str(dtype), phase_code)
        index_sets.setdefault(identity, set()).add(operation_index)
        pair_key = (*identity, iteration, operation_index)
        by_rank = pair_rows.setdefault(pair_key, {})
        if rank in by_rank:
            raise CoverageError(
                f"duplicate duration for rank {rank}, iteration {iteration}, operation_index {operation_index}"
            )
        by_rank[rank] = float(duration)

    if not index_sets:
        raise CoverageError("no stable tensor_parallel attention_backward rows")
    selected: dict[tuple[Any, ...], set[int]] = {}
    for identity, indices in index_sets.items():
        segments = _contiguous_runs(indices)
        if (len(segments) != FIRST_PHASE_BLOCK_COUNT or
                any(len(block) != FIRST_PHASE_BLOCK_SIZE for block in segments)):
            raise CoverageError(
                "attention_backward operation_index coverage is incomplete: "
                f"expected two contiguous blocks of three, found {[len(block) for block in segments]}"
            )
        selected[identity] = {block[0] for block in segments}

    waits_by_rank_phase: dict[tuple[int, int], list[float]] = {}
    for identity, first_indices in selected.items():
        members, _label, _name, _kind, _count, _dtype, phase_code = identity
        for pair_key, by_rank in pair_rows.items():
            if pair_key[:len(identity)] != identity or pair_key[-1] not in first_indices:
                continue
            if set(by_rank) != set(members):
                raise CoverageError(
                    f"selected operation_index {pair_key[-1]} is not paired across TP members"
                )
            for rank in members:
                peer = members[1] if rank == members[0] else members[0]
                waits_by_rank_phase.setdefault((rank, phase_code), []).append(
                    by_rank[peer] - by_rank[rank]
                )

    # The configured Qwen 24-layer / PP2 scope must produce 12 distinct,
    # two-microbatch phase scores for every one of the 16 ranks.
    all_phase_codes: set[int] = set()
    phase_scores: dict[int, dict[int, float]] = {rank: {} for rank in EXPECTED_RANKS}
    for rank in EXPECTED_RANKS:
        rank_codes = sorted(code for candidate_rank, code in waits_by_rank_phase if candidate_rank == rank)
        if len(rank_codes) != PHASES_PER_RANK or len(set(rank_codes)) != PHASES_PER_RANK:
            raise CoverageError(
                f"rank {rank} has {len(set(rank_codes))} attention_backward phases; "
                f"expected {PHASES_PER_RANK}"
            )
        for phase_code in rank_codes:
            values = waits_by_rank_phase[(rank, phase_code)]
            if len(values) != FIRST_PHASE_BLOCK_COUNT:
                raise CoverageError(
                    f"rank {rank} phase {phase_code} has {len(values)} paired microbatch waits; "
                    f"expected {FIRST_PHASE_BLOCK_COUNT}"
                )
            phase_scores[rank][phase_code] = float(statistics.median(values))
            all_phase_codes.add(phase_code)
    if len(all_phase_codes) != PHASES_PER_RANK * PIPELINE_STAGES:
        raise CoverageError(
            f"observed {len(all_phase_codes)} global attention_backward phases; "
            f"expected {TOTAL_LAYERS}"
        )

    rank_scores = {
        rank: float(statistics.median(phase_scores[rank].values()))
        for rank in EXPECTED_RANKS
    }
    observed_rank_phases = sum(len(values) for values in phase_scores.values())
    observed_microbatch_pairs = sum(
        len(values) for (rank, _code), values in waits_by_rank_phase.items()
    )
    hashes = {
        "manifest.json": _sha256_file(run_dir / "manifest.json"),
        "topology.json": _sha256_file(topology_path),
        "profile.json": _sha256_file(profile_path),
    }
    return RunScores(
        run_id=run_dir.name,
        rank_scores_us=rank_scores,
        phase_scores_us=phase_scores,
        coverage={
            "complete": True,
            "observed_ranks": len(rank_scores),
            "expected_ranks": WORLD_SIZE,
            "observed_rank_phases": observed_rank_phases,
            "expected_rank_phases": WORLD_SIZE * PHASES_PER_RANK,
            "observed_microbatch_pairs": observed_microbatch_pairs,
            "expected_microbatch_pairs": WORLD_SIZE * PHASES_PER_RANK * FIRST_PHASE_BLOCK_COUNT,
        },
        source_hashes=hashes,
    )


def score(path: Path | str) -> RunScores:
    """Score one run using paired first-call waiting durations."""
    return _extract_wait_scores(Path(path))


def _source_code_hash() -> str:
    return _sha256_file(Path(__file__))


def fit_dataset(dataset: Path | str) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError(f"dataset directory does not exist: {dataset}")
    selected: list[tuple[Path, dict[str, Any]]] = []
    session_roles: dict[str, set[str]] = {}
    for run_dir in sorted((p for p in dataset.iterdir() if p.is_dir()), key=lambda p: p.name):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_run_manifest(run_dir)
        role = manifest.get("role")
        session_id = manifest.get("session_id", manifest.get("session"))
        if role not in {"train", "validation", "blind_test"}:
            raise ValueError(f"{manifest_path}: invalid role")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{manifest_path}: session_id is required")
        session_roles.setdefault(session_id, set()).add(role)
        if role in {"train", "validation"}:
            selected.append((run_dir, manifest))
    leaked = {session: sorted(roles) for session, roles in session_roles.items() if len(roles) > 1}
    if leaked:
        raise ValueError(f"session split leakage: {leaked}")

    # Read fault_rank only to select healthy calibration episodes. Fault ranks
    # are not included in features, thresholds, source records, or the model.
    healthy = [
        (run_dir, manifest) for run_dir, manifest in selected
        if manifest.get("fault_rank") is None
    ]
    train_healthy = [(p, m) for p, m in healthy if m.get("role") == "train"]
    validation_healthy = [(p, m) for p, m in healthy if m.get("role") == "validation"]
    if not train_healthy or not validation_healthy:
        raise ValueError("fit requires healthy train and healthy validation runs")
    train_sessions = {str(m.get("session_id", m.get("session"))) for _, m in train_healthy}
    if len(train_sessions) < 2:
        raise ValueError("healthy calibration requires train runs from at least two sessions")

    scored = []
    source_runs = []
    for run_dir, manifest in healthy:
        result = _extract_wait_scores(run_dir, manifest=manifest)
        scored.append((manifest["role"], result))
        source_runs.append({
            "run_id": run_dir.name,
            "session_id": manifest.get("session_id", manifest.get("session")),
            "role": manifest["role"],
            "files": result.source_hashes,
        })
    positive_scores = [
        value for _role, run in scored for value in run.rank_scores_us.values()
        if value > 0
    ]
    if not positive_scores:
        raise ValueError(
            "healthy calibration H is zero; require a positive healthy waiting score; "
            "the explicit score-scale floor is 1.0 us"
        )
    measured_h = max(positive_scores)
    scale_h = max(measured_h, SCORE_FLOOR_US)
    thresholds = {
        "healthy_positive_max_us": measured_h,
        "scale_h_us": scale_h,
        "healthy_limit_us": HEALTHY_LIMIT_MULTIPLIER * scale_h,
        "fault_limit_us": FAULT_LIMIT_MULTIPLIER * scale_h,
        "healthy_limit_multiplier": HEALTHY_LIMIT_MULTIPLIER,
        "fault_limit_multiplier": FAULT_LIMIT_MULTIPLIER,
        "minimum_top_to_runner_up_ratio": MIN_TOP_RATIO,
        "ratio_denominator_floor_us": SCORE_FLOOR_US,
        "healthy_runs_used": len(scored),
        "healthy_train_runs_used": len(train_healthy),
        "healthy_validation_runs_used": len(validation_healthy),
    }
    source_payload = [
        {"run_id": row["run_id"], "session_id": row["session_id"],
         "role": row["role"], "files": row["files"]}
        for row in sorted(source_runs, key=lambda item: item["run_id"])
    ]
    model: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "algorithm": "paired_first_call_signed_wait_localization",
        "description": (
            "Interpretable FLASH late-arriving communication extension; "
            "not the original paper algorithm."
        ),
        "feature_spec": {
            "score": "peer_duration_us - rank_duration_us",
            "pair_key": ["iteration", "phase_code", "operation_index", "TP members"],
            "microbatch_aggregation": "median of two verified first-call pairs",
            "layer_aggregation": "median across 12 local attention-backward phases",
            "identity": ["TP members", "semantic topology group/tag", "phase_code", "kind", "count", "dtype"],
            "excluded": ["fault_rank", "fault logs", "declared delay", "clique", "group_id", "timestamp_us"],
            "coverage": "all 16 ranks, 12 phases per rank, 2 paired microbatch calls per phase",
        },
        "config": {
            "world_size": WORLD_SIZE,
            "tp": TENSOR_PARALLEL_SIZE,
            "pp": PIPELINE_STAGES,
            "total_layers": TOTAL_LAYERS,
            "phases_per_rank": PHASES_PER_RANK,
            "expected_blocks": FIRST_PHASE_BLOCK_COUNT,
            "calls_per_block": FIRST_PHASE_BLOCK_SIZE,
            "sampling_stride": FIRST_PHASE_SAMPLING_STRIDE,
            "score_floor_us": SCORE_FLOOR_US,
        },
        "thresholds": thresholds,
        "healthy_calibration_counts": {
            "runs": len(scored),
            "train_runs": len(train_healthy),
            "validation_runs": len(validation_healthy),
            "sessions": sorted({row["session_id"] for row in source_runs}),
            "ranks_per_run": WORLD_SIZE,
            "phases_per_rank": PHASES_PER_RANK,
        },
        "source_runs": source_runs,
        "source_hash": _sha256_bytes(_canonical_json(source_payload).encode("utf-8")),
        "source_code_hash": _source_code_hash(),
    }
    model["frozen_model_hash"] = _sha256_bytes(_canonical_json(model).encode("utf-8"))
    return model


def verify_model(model: dict[str, Any]) -> None:
    if model.get("model_version") != MODEL_VERSION:
        raise ValueError(f"unsupported wait_localize model version: {model.get('model_version')}")
    frozen = model.get("frozen_model_hash")
    payload = dict(model)
    payload.pop("frozen_model_hash", None)
    if frozen != _sha256_bytes(_canonical_json(payload).encode("utf-8")):
        raise ValueError("frozen model hash mismatch")
    if model.get("source_code_hash") != _source_code_hash():
        raise ValueError("wait_localize source code hash mismatch; refit with this implementation")


def localize_scores(scores: dict[int, float], thresholds: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        ((int(rank), float(value)) for rank, value in scores.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    if set(scores) != set(EXPECTED_RANKS) or any(not math.isfinite(value) for _, value in ranked):
        return {
            "predicted": "inconclusive",
            "reason": "incomplete_rank_coverage",
            "top_rank": None,
            "top_score_us": None,
            "runner_up_score_us": None,
            "margin_us": None,
            "top_to_runner_up_ratio": None,
        }
    nonnegative = [pair for pair in ranked if pair[1] >= 0.0]
    if not nonnegative:
        return {
            "predicted": "healthy",
            "reason": "all_rank_scores_nonpositive",
            "top_rank": None,
            "top_score_us": max(value for _, value in ranked),
            "runner_up_score_us": None,
            "margin_us": None,
            "top_to_runner_up_ratio": None,
        }
    top_rank, top_score = nonnegative[0]
    runner_up_score = nonnegative[1][1] if len(nonnegative) > 1 else 0.0
    margin = top_score - runner_up_score
    ratio = top_score / max(runner_up_score, float(thresholds["ratio_denominator_floor_us"]))
    healthy_limit = float(thresholds["healthy_limit_us"])
    fault_limit = float(thresholds["fault_limit_us"])
    if top_score <= healthy_limit:
        predicted, reason = "healthy", "within_healthy_limit"
    elif top_score < fault_limit:
        predicted, reason = "inconclusive", "hysteresis_band"
    elif ratio < float(thresholds["minimum_top_to_runner_up_ratio"]):
        predicted, reason = "inconclusive", "top_runner_up_ratio_too_small"
    else:
        predicted, reason = f"rank:{top_rank}", "accepted_top_signed_wait"
    return {
        "predicted": predicted,
        "reason": reason,
        "top_rank": top_rank,
        "top_score_us": top_score,
        "runner_up_score_us": runner_up_score,
        "margin_us": margin,
        "top_to_runner_up_ratio": ratio,
    }


def localize_run(run_dir: Path | str, model: dict[str, Any]) -> dict[str, Any]:
    verify_model(model)
    result = score(run_dir)
    decision = localize_scores(result.rank_scores_us, model["thresholds"])
    return {
        "run_id": result.run_id,
        "frozen_model_hash": model["frozen_model_hash"],
        "source_hashes": result.source_hashes,
        "coverage": result.coverage,
        "rank_scores_us": result.rank_scores_us,
        "phase_scores_us": result.phase_scores_us,
        **decision,
    }


def evaluate_dataset(
    dataset: Path | str,
    model: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    verify_model(model)
    if split not in {"validation", "blind_test"}:
        raise ValueError("split must be validation or blind_test")
    dataset = Path(dataset).resolve()
    selected: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in sorted((p for p in dataset.iterdir() if p.is_dir()), key=lambda p: p.name):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_run_manifest(run_dir)
        if manifest.get("role") == split:
            selected.append((run_dir, manifest))
    if not selected:
        raise ValueError(f"dataset has no {split} runs")
    if split == "blind_test":
        fit_sessions = {str(row["session_id"]) for row in model["source_runs"]}
        eval_sessions = {
            str(manifest.get("session_id", manifest.get("session")))
            for _path, manifest in selected
        }
        overlap = sorted(fit_sessions & eval_sessions)
        if overlap:
            raise ValueError(f"blind_test sessions overlap healthy fit sources: {overlap}")

    runs = []
    for run_dir, manifest in selected:
        try:
            result = _extract_wait_scores(run_dir, manifest=manifest)
            decision = localize_scores(result.rank_scores_us, model["thresholds"])
            item = {
                "run_id": result.run_id,
                "session_id": manifest.get("session_id", manifest.get("session")),
                "coverage": result.coverage,
                "rank_scores_us": result.rank_scores_us,
                "phase_scores_us": result.phase_scores_us,
                "source_hashes": result.source_hashes,
                **decision,
            }
        except (CoverageError, FeatureSelectionUnavailable, ValueError, TypeError, KeyError) as error:
            item = {
                "run_id": run_dir.name,
                "session_id": manifest.get("session_id", manifest.get("session")),
                "coverage": {"complete": False},
                "rank_scores_us": {},
                "phase_scores_us": {},
                "predicted": "inconclusive",
                "reason": "incomplete_or_invalid_coverage",
                "error": str(error),
            }
        # Truth enters only after this run's score and threshold decision exist.
        label = manifest.get("fault_rank")
        if label is not None and (type(label) is not int or label < 0):
            raise ValueError(f"{run_dir.name}: fault_rank must be a nonnegative integer or null")
        item["true_label"] = "healthy" if label is None else f"rank:{label}"
        runs.append(item)

    known_labels = {"healthy", *(f"rank:{rank}" for rank in EXPECTED_RANKS)}
    predicted_labels = known_labels | {"inconclusive"}
    true_labels = sorted(known_labels | {row["true_label"] for row in runs},
                         key=lambda value: (value != "healthy", value))
    prediction_columns = sorted(predicted_labels, key=lambda value: (value != "healthy", value))
    matrix = {label: {prediction: 0 for prediction in prediction_columns} for label in true_labels}
    for item in runs:
        matrix[item["true_label"]][item["predicted"]] += 1
    healthy_runs = [row for row in runs if row["true_label"] == "healthy"]
    inconclusive = sum(row["predicted"] == "inconclusive" for row in runs)
    correct = sum(row["true_label"] == row["predicted"] for row in runs)
    decided = len(runs) - inconclusive
    healthy_fp = sum(row["predicted"].startswith("rank:") for row in healthy_runs)
    return {
        "report_version": 1,
        "method": "paired_first_call_signed_wait_localization",
        "split": split,
        "frozen_model_hash": model["frozen_model_hash"],
        "evaluation_source_hash": _sha256_bytes(_canonical_json([
            {"run_id": row["run_id"], "source_hashes": row.get("source_hashes", {})}
            for row in runs
        ]).encode("utf-8")),
        "run_accuracy": correct / len(runs) if runs else None,
        "correct_runs": correct,
        "run_denominator": len(runs),
        "decided_accuracy": correct / decided if decided else None,
        "decided_run_denominator": decided,
        "inconclusive_runs": inconclusive,
        "inconclusive_denominator": len(runs),
        "healthy_false_positives": healthy_fp,
        "healthy_false_positive_denominator": len(healthy_runs),
        "confusion_matrix": matrix,
        "runs": runs,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="calibrate limits from healthy train and validation runs")
    fit.add_argument("--dataset", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    localize = subparsers.add_parser("localize", help="localize one run with a frozen model")
    localize.add_argument("--run", required=True, type=Path)
    localize.add_argument("--model", required=True, type=Path)
    localize.add_argument("--output", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen model on validation or blind_test")
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--split", required=True, choices=("validation", "blind_test"))
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fit":
            model = fit_dataset(args.dataset)
            _write_json(args.output, model)
            print(json.dumps({
                "output": str(args.output),
                "frozen_model_hash": model["frozen_model_hash"],
                "source_hash": model["source_hash"],
                "healthy_limit_us": model["thresholds"]["healthy_limit_us"],
                "fault_limit_us": model["thresholds"]["fault_limit_us"],
                "healthy_runs_used": model["thresholds"]["healthy_runs_used"],
            }, indent=2, ensure_ascii=False))
        elif args.command == "localize":
            model = _read_json(args.model)
            report = localize_run(args.run, model)
            if args.output:
                _write_json(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        else:
            model = _read_json(args.model)
            report = evaluate_dataset(args.dataset, model, split=args.split)
            if args.output:
                _write_json(args.output, report)
            print(json.dumps({
                "split": args.split,
                "run_accuracy": report["run_accuracy"],
                "correct_runs": report["correct_runs"],
                "run_denominator": report["run_denominator"],
                "inconclusive_runs": report["inconclusive_runs"],
                "healthy_false_positives": report["healthy_false_positives"],
            }, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"wait_localize: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

