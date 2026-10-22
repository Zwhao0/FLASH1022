"""Session-split, run-level offline communication-fingerprint research tool.

Only stable, topology-labeled communication rows are inputs. Manifest fault
labels are used by the scorer after features and predictions have been made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


MODEL_VERSION = 3
ALLOWED_ROLES = {"train", "validation", "blind_test"}
ALLOWED_CALL_SELECTIONS = {"all", "first_in_phase"}
DEFAULT_MIN_COVERAGE = 0.70
DEFAULT_SCALE_FLOOR = 0.05
FIRST_PHASE_BLOCK_SIZE = 3
FIRST_PHASE_BLOCK_COUNT = 2
FIRST_PHASE_SAMPLING_STRIDE = 8


class FeatureSelectionUnavailable(ValueError):
    """Required operation-index evidence is missing or incomplete."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def _scalar(value: Any) -> Any:
    """Keep stable JSON scalars as feature fields; normalize exotic values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite feature scalar")
        return value
    return str(value)


def _feature_definition(
    members: tuple[int, ...],
    label: str,
    group_name: str,
    kind: Any,
    count: Any,
    dtype: Any,
    phase_code: int,
    rank: int,
) -> dict[str, Any]:
    definition = {
        "members": list(members),
        "semantic_label": label,
        "group_name": group_name,
        "kind": _scalar(kind),
        "count": _scalar(count),
        "dtype": _scalar(dtype),
        "phase_code": phase_code,
        "rank": rank,
    }
    definition["id"] = _canonical_json(definition)
    return definition


def _group_is_stable(group: dict[str, Any]) -> bool:
    # The current topology producer uses localization; accept an explicit stable
    # marker too, while treating absent or contradictory markers as unstable.
    markers = [group[key] for key in ("localization", "stable") if key in group]
    if any(marker is False for marker in markers):
        return False
    if any(marker is True for marker in markers):
        return True
    return group.get("stability") in {"stable", "localization"}


def _stable_groups(topology: Any) -> dict[tuple[tuple[int, ...], str, str], dict[str, Any]]:
    if not isinstance(topology, dict) or not isinstance(topology.get("groups"), list):
        raise ValueError("topology.json must contain a groups list")
    world_size = topology.get("world_size")
    stable: dict[tuple[tuple[int, ...], str, str], dict[str, Any]] = {}
    for group in topology["groups"]:
        if not isinstance(group, dict) or not _group_is_stable(group):
            continue
        members_value = group.get("members")
        if (not isinstance(members_value, list) or len(members_value) < 2 or
                any(type(rank) is not int or rank < 0 for rank in members_value) or
                len(set(members_value)) != len(members_value)):
            continue
        members = tuple(sorted(members_value))
        if isinstance(world_size, int) and any(rank >= world_size for rank in members):
            continue
        reporters = group.get("reporters")
        if reporters is not None and (not isinstance(reporters, list) or
                                      set(reporters) != set(members)):
            continue
        name = group.get("name")
        label = group.get("label", group.get("semantic_label", group.get("tag")))
        if not isinstance(name, str) or not name or not isinstance(label, str) or not label:
            continue
        key = (members, label, name)
        # Ambiguous duplicate identities are excluded rather than guessed.
        if key in stable and stable[key] != group:
            stable.pop(key, None)
            continue
        stable[key] = group
    return stable


def _profile_rows(profile: Any) -> list[dict[str, Any]]:
    if isinstance(profile, list):
        rows = profile
    elif isinstance(profile, dict):
        rows = profile.get("rows")
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("profile.json must be a row list or an object with rows")
    return [row for row in rows if isinstance(row, dict)]


_PHASE_SUBCODES = {
    "attn_fwd": 1,
    "attention_forward": 1,
    "attn_bwd": 2,
    "attention_backward": 2,
    "mlp_fwd": 3,
    "mlp_forward": 3,
    "mlp_bwd": 4,
    "mlp_backward": 4,
}


def _row_phase_code(row: dict[str, Any]) -> int | None:
    code = row.get("phase_code")
    if code is not None:
        if type(code) is int and code >= 0:
            return code
        return None
    layer = row.get("global_layer", row.get("layer"))
    phase = row.get("phase")
    if layer is None and phase is None:
        return 0
    if type(layer) is not int or layer < 0 or not isinstance(phase, str):
        return None
    subcode = _PHASE_SUBCODES.get(phase.strip().lower())
    if subcode is None:
        return None
    return ((layer + 1) << 8) | subcode


def _phase_selected(phase_code: int, phases: str) -> bool:
    if phases == "all":
        return True
    if phases == "attention_backward":
        return (phase_code & 0xFF) == 2
    raise ValueError(f"unsupported phase filter: {phases}")


def _is_tensor_parallel(group_name: str) -> bool:
    """Use the topology's semantic group name, not its process-group tag."""
    return group_name.strip().lower() == "tensor_parallel"


def _has_phase_metadata(row: dict[str, Any]) -> bool:
    return row.get("phase_code") is not None or (
        row.get("global_layer", row.get("layer")) is not None and row.get("phase") is not None
    )


def _contiguous_runs(indices: Iterable[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(set(indices)):
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def extract_run_features_data(
    topology: Any,
    profile: Any,
    *,
    phases: str = "all",
    call_selection: str = "all",
) -> dict[str, float]:
    """Return per-run median peer-centered log durations keyed by stable identity.

    The sign is ``median(peer log duration) - this rank's log duration``. The
    feature key includes semantic phase_code. Clique ids, group ids, timestamps,
    run ids, and fault labels never enter a key.
    """
    if call_selection not in ALLOWED_CALL_SELECTIONS:
        raise ValueError(f"call_selection must be one of {sorted(ALLOWED_CALL_SELECTIONS)}")
    if call_selection == "first_in_phase" and phases != "attention_backward":
        raise ValueError("first_in_phase requires phases=attention_backward")
    stable_groups = _stable_groups(topology)
    # Parse once so index selection is based on the complete run, before any
    # per-iteration median can combine a phase's first and later collectives.
    parsed_rows = []
    indices_by_group: dict[tuple[Any, ...], set[int]] = {}
    ranks_by_index: dict[tuple[tuple[Any, ...], int], set[int]] = {}
    rows = _profile_rows(profile)
    for row in rows:
        rank = row.get("rank")
        members_value = row.get("group_members")
        group_name = row.get("group_name")
        tag = row.get("tag")
        if not isinstance(group_name, str) or not isinstance(tag, str):
            continue
        if not isinstance(members_value, list) or any(type(member) is not int for member in members_value):
            if call_selection == "first_in_phase" and _is_tensor_parallel(group_name):
                raise FeatureSelectionUnavailable("tensor_parallel row lacks actual group members")
            continue
        members = tuple(sorted(members_value))
        group = stable_groups.get((members, tag, group_name))
        if group is None:
            # A row tag may contain a suffix for operation metadata. It is
            # accepted only when it still names the topology's semantic label.
            candidates = [candidate for (candidate_members, label, name), candidate in stable_groups.items()
                          if candidate_members == members and name == group_name and
                          tag.startswith(label + ":")]
            if len(candidates) != 1:
                continue
            group = candidates[0]
            semantic_label = group.get("label", group.get("semantic_label", group.get("tag")))
        else:
            semantic_label = tag
        semantic_label = str(semantic_label)
        if call_selection == "first_in_phase":
            # The topology match above establishes the actual semantic group.
            # Row tags such as TP_PP0/TP_PP1 identify PP-specific instances,
            # not the tensor-parallel group kind.
            if not _is_tensor_parallel(str(group.get("name", ""))):
                continue
            if len(members) != 2:
                raise FeatureSelectionUnavailable("first_in_phase expects tensor_parallel groups with two members")
            if not _has_phase_metadata(row):
                raise FeatureSelectionUnavailable("tensor_parallel phase metadata is required for first_in_phase")

        phase_code = _row_phase_code(row)
        if phase_code is None:
            if call_selection == "first_in_phase":
                raise FeatureSelectionUnavailable("invalid phase metadata in tensor_parallel rows")
            continue
        if not _phase_selected(phase_code, phases):
            continue
        if call_selection == "first_in_phase" and (phase_code & 0xFF) != 2:
            continue

        iteration = row.get("iteration")
        duration = row.get("duration_us")
        count = row.get("count")
        dtype = row.get("dtype")
        kind = row.get("kind", row.get("operation", group.get("kind", "unspecified")))
        if type(rank) is not int or rank not in members or type(iteration) is not int or iteration < 0:
            if call_selection == "first_in_phase":
                raise FeatureSelectionUnavailable("tensor_parallel phase row lacks rank or iteration")
            continue
        if count is None or dtype is None:
            if call_selection == "first_in_phase":
                raise FeatureSelectionUnavailable("tensor_parallel phase row lacks count or dtype")
            continue
        if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0:
            continue
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        if not math.isfinite(float(duration)) or float(duration) <= 0:
            continue

        group_identity = (
            members, semantic_label, group_name, _scalar(kind), _scalar(count),
            _scalar(dtype), phase_code,
        )
        operation_index = None
        if call_selection == "first_in_phase":
            operation_index = row.get("operation_index")
            if type(operation_index) is not int or operation_index < 0:
                raise FeatureSelectionUnavailable("operation_index is required for first_in_phase")
            indices_by_group.setdefault(group_identity, set()).add(operation_index)
            ranks_by_index.setdefault((group_identity, operation_index), set()).add(rank)
        parsed_rows.append({
            "rank": rank,
            "iteration": iteration,
            "duration_log": math.log(float(duration)),
            "group_identity": group_identity,
            "operation_index": operation_index,
        })

    selected_first_calls: dict[tuple[Any, ...], set[int]] = {}
    if call_selection == "first_in_phase":
        if not indices_by_group:
            raise FeatureSelectionUnavailable("no complete tensor_parallel attention_backward call indices")
        expected_groups = {
            (members, label, name)
            for (members, label, name), group in stable_groups.items()
            if _is_tensor_parallel(str(group.get("name", ""))) and len(members) == 2
        }
        observed_groups = {identity[:3] for identity in indices_by_group}
        missing_groups = sorted(expected_groups - observed_groups, key=lambda item: (item[1], item[2], item[0]))
        if missing_groups:
            raise FeatureSelectionUnavailable(
                f"tensor_parallel attention_backward groups lack complete profile rows: {missing_groups}"
            )
        for group_identity, indices in indices_by_group.items():
            members = set(group_identity[0])
            for operation_index in indices:
                observed = ranks_by_index.get((group_identity, operation_index), set())
                if observed != members:
                    raise FeatureSelectionUnavailable(
                        f"operation_index {operation_index} is missing tensor_parallel members"
                    )
            segments = _contiguous_runs(indices)
            if (len(segments) != FIRST_PHASE_BLOCK_COUNT or
                    any(len(segment) != FIRST_PHASE_BLOCK_SIZE for segment in segments)):
                raise FeatureSelectionUnavailable(
                    "attention_backward operation_index coverage is incomplete: "
                    f"expected {FIRST_PHASE_BLOCK_COUNT} contiguous blocks of "
                    f"{FIRST_PHASE_BLOCK_SIZE}, found {[len(segment) for segment in segments]}"
                )
            selected_first_calls[group_identity] = {segment[0] for segment in segments}

    # identity + iteration -> per-rank log-duration samples
    per_iteration: dict[tuple[Any, ...], dict[int, list[float]]] = {}
    for row in parsed_rows:
        group_identity = row["group_identity"]
        if call_selection == "first_in_phase" and row["operation_index"] not in selected_first_calls[group_identity]:
            continue
        identity = (*group_identity, row["iteration"])
        by_rank = per_iteration.setdefault(identity, {})
        by_rank.setdefault(row["rank"], []).append(row["duration_log"])

    feature_samples: dict[str, list[float]] = {}
    for identity, rank_samples in per_iteration.items():
        rank_logs = {rank: _median(values) for rank, values in rank_samples.items()}
        if len(rank_logs) < 2:
            continue
        members, label, group_name, kind, count, dtype, phase_code, _iteration = identity
        for rank, own_log in rank_logs.items():
            peer_logs = [value for peer_rank, value in rank_logs.items() if peer_rank != rank]
            if not peer_logs:
                continue
            definition = _feature_definition(members, label, group_name, kind, count, dtype, phase_code, rank)
            feature_samples.setdefault(definition["id"], []).append(_median(peer_logs) - own_log)
    return {feature_id: _median(values) for feature_id, values in feature_samples.items()}


def extract_run_features(
    topology_path: Path | str,
    profile_path: Path | str,
    *,
    phases: str = "all",
    call_selection: str = "all",
) -> dict[str, float]:
    return extract_run_features_data(
        _read_json(Path(topology_path)),
        _read_json(Path(profile_path)),
        phases=phases,
        call_selection=call_selection,
    )


def _session_id(manifest: dict[str, Any]) -> str | None:
    session = manifest.get("session_id", manifest.get("session"))
    if session is None and isinstance(manifest.get("config"), dict):
        session = manifest["config"].get("session_id")
    return str(session) if session is not None and str(session) else None


def _label_from_manifest(manifest: dict[str, Any]) -> int | None:
    label = manifest.get("fault_rank")
    if label is None:
        return None
    if type(label) is not int or label < 0:
        raise ValueError("manifest fault_rank must be a nonnegative integer or null")
    return label


def _validate_first_phase_config(manifest: dict[str, Any]) -> None:
    config = manifest.get("configuration", manifest.get("config"))
    if not isinstance(config, dict):
        raise FeatureSelectionUnavailable("first_in_phase requires manifest configuration metadata")
    expected = {
        "attention_backward_block_size": FIRST_PHASE_BLOCK_SIZE,
        "microbatches": FIRST_PHASE_BLOCK_COUNT,
    }
    for name, expected_value in expected.items():
        value = config.get(name)
        if type(value) is not int or value != expected_value:
            raise FeatureSelectionUnavailable(
                f"first_in_phase requires manifest {name}={expected_value}; got {value!r}"
            )
    sampling_stride = config.get("sampling_stride")
    sample_stride = config.get("sample_stride")
    if (sampling_stride is not None and sample_stride is not None and
            (type(sampling_stride) is not int or type(sample_stride) is not int or
             sampling_stride != sample_stride)):
        raise FeatureSelectionUnavailable(
            "first_in_phase requires matching sampling_stride and sample_stride aliases"
        )
    value = sampling_stride if sampling_stride is not None else sample_stride
    if type(value) is not int or value != FIRST_PHASE_SAMPLING_STRIDE:
        raise FeatureSelectionUnavailable(
            "first_in_phase requires manifest sampling_stride=8 or sample_stride=8; "
            f"got {value!r}"
        )


def _catalog_runs(dataset: Path, *, load_labels_for: set[str]) -> list[dict[str, Any]]:
    if not dataset.is_dir():
        raise ValueError(f"dataset directory does not exist: {dataset}")
    catalog = []
    for run_dir in sorted((path for path in dataset.iterdir() if path.is_dir()), key=lambda path: path.name):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"manifest must be an object: {manifest_path}")
        role = manifest.get("role")
        session_id = _session_id(manifest)
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{manifest_path}: role must be one of {sorted(ALLOWED_ROLES)}")
        if session_id is None:
            raise ValueError(f"{manifest_path}: add session_id so episodes cannot cross splits")
        catalog.append({
            "run_id": run_dir.name,
            "run_dir": run_dir,
            "manifest_path": manifest_path,
            "role": role,
            "session_id": session_id,
            # Labels are only consumed for selected train/validation runs or by
            # the final scorer; held-out labels are not retained by the loader.
            "manifest": manifest if role in load_labels_for else None,
        })
    if not catalog:
        raise ValueError(f"no run directories with manifest.json found under {dataset}")
    roles_by_session: dict[str, set[str]] = {}
    for run in catalog:
        roles_by_session.setdefault(run["session_id"], set()).add(run["role"])
    leaked = {session: sorted(roles) for session, roles in roles_by_session.items() if len(roles) != 1}
    if leaked:
        raise ValueError(f"session split leakage: each session must have one role: {leaked}")
    return catalog


def _load_run(
    run: dict[str, Any],
    *,
    phases: str = "all",
    call_selection: str = "all",
) -> dict[str, Any]:
    topology_path = run["run_dir"] / "topology.json"
    profile_path = run["run_dir"] / "profile.json"
    if not topology_path.is_file() or not profile_path.is_file():
        raise ValueError(f"{run['run_id']}: expected topology.json and profile.json")
    feature_error = None
    try:
        if call_selection == "first_in_phase":
            _validate_first_phase_config(run["manifest"])
        features = extract_run_features(
            topology_path, profile_path, phases=phases, call_selection=call_selection
        )
    except FeatureSelectionUnavailable as error:
        features = {}
        feature_error = str(error)
    hashes = {
        "manifest.json": _sha256_file(run["manifest_path"]),
        "topology.json": _sha256_file(topology_path),
        "profile.json": _sha256_file(profile_path),
    }
    manifest = run["manifest"]
    return {
        "run_id": run["run_id"],
        "session_id": run["session_id"],
        "role": run["role"],
        "label": _label_from_manifest(manifest),
        "features": features,
        "feature_error": feature_error,
        "source_hashes": hashes,
    }


def _source_records(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run["run_id"],
            "session_id": run["session_id"],
            "role": run["role"],
            "fault_rank_label": run["label"],
            "files": run["source_hashes"],
        }
        for run in sorted(runs, key=lambda item: item["run_id"])
    ]


def _robust_center_scale(values: list[float], floor: float) -> tuple[float, float]:
    center = _median(values)
    mad = _median(abs(value - center) for value in values)
    return center, max(1.4826 * mad, floor)


def _normalized_features(features: dict[str, float], baselines: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        feature_id: (value - baselines[feature_id]["center"]) / baselines[feature_id]["scale"]
        for feature_id, value in features.items()
        if feature_id in baselines
    }


def _rms(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.inf
    return math.sqrt(sum(value * value for value in values) / len(values))


def _template_scores(
    z_features: dict[str, float],
    templates: dict[str, list[dict[str, Any]]],
    expected_count: int,
    min_coverage: float,
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for label, class_templates in templates.items():
        best: dict[str, Any] | None = None
        for template in class_templates:
            vector = template["values"]
            common = sorted(set(z_features) & set(vector))
            pair_coverage = len(common) / expected_count if expected_count else 0.0
            if pair_coverage < min_coverage:
                continue
            distance = _rms(z_features[key] - vector[key] for key in common)
            candidate = {
                "distance": distance,
                "coverage": pair_coverage,
                "template_run_id": template["run_id"],
            }
            if best is None or (distance, template["run_id"]) < (best["distance"], best["template_run_id"]):
                best = candidate
        if best is not None:
            scores[label] = best
    return scores


def _score_margin(scores: dict[str, dict[str, Any]]) -> tuple[str | None, float, float]:
    ordered = sorted(scores.items(), key=lambda item: (item[1]["distance"], int(item[0])))
    if not ordered:
        return None, math.inf, 0.0
    best_label, best_score = ordered[0]
    if len(ordered) == 1:
        return best_label, best_score["distance"], math.inf
    return best_label, best_score["distance"], ordered[1][1]["distance"] - best_score["distance"]


def _run_diagnostics(
    run: dict[str, Any],
    model: dict[str, Any],
    *,
    apply_thresholds: bool = True,
) -> dict[str, Any]:
    baselines = model["healthy_baselines"]
    feature_ids = model["feature_ids"]
    min_coverage = model["config"]["min_coverage"]
    z_features = _normalized_features(run["features"], baselines)
    expected_count = len(feature_ids)
    coverage = len(z_features) / expected_count if expected_count else 0.0
    missing = sorted(set(feature_ids) - set(z_features))
    anomaly = _rms(z_features.values()) if z_features else None
    scores = _template_scores(z_features, model["templates"], expected_count, min_coverage)
    best_label, distance, margin = _score_margin(scores)
    details: dict[str, Any] = {
        "run_id": run["run_id"],
        "session_id": run["session_id"],
        "coverage": coverage,
        "observed_feature_count": len(z_features),
        "expected_feature_count": expected_count,
        "missing_feature_count": len(missing),
        "missing_feature_ids": missing,
        "healthy_anomaly_score": anomaly,
        "class_scores": scores,
        "best_class": int(best_label) if best_label is not None else None,
        "best_distance": distance if best_label is not None else None,
        "class_margin": margin if math.isfinite(margin) else None,
        "predicted_label": "inconclusive",
        "reason": "unresolved",
        "feature_error": run.get("feature_error"),
    }
    if not apply_thresholds:
        return details
    thresholds = model["thresholds"]
    if details["feature_error"]:
        details["reason"] = "call_selection_unavailable"
        return details
    if coverage < min_coverage:
        details["reason"] = "insufficient_feature_coverage"
        return details
    if thresholds.get("healthy_max_anomaly") is None:
        details["reason"] = "healthy_threshold_unavailable"
        return details
    if anomaly <= thresholds["healthy_max_anomaly"]:
        details["predicted_label"] = None
        details["reason"] = "within_healthy_validation_envelope"
        return details
    if not scores:
        details["reason"] = "insufficient_template_overlap"
        return details
    if thresholds.get("max_fault_distance") is None:
        details["reason"] = "no_validation_fault_template_accepted"
        return details
    if len(model["templates"]) > 1 and margin < thresholds["min_class_margin"]:
        details["reason"] = "classes_too_close"
        return details
    if distance > thresholds["max_fault_distance"]:
        details["reason"] = "outside_validation_template_distance"
        return details
    if best_label not in thresholds.get("calibrated_labels", []):
        details["reason"] = "class_not_separable_on_validation"
        return details
    details["predicted_label"] = int(best_label)
    details["reason"] = "accepted_nearest_template"
    return details


def _calibrate_thresholds(validation_runs: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    min_coverage = model["config"]["min_coverage"]
    eligible_healthy_anomalies = []
    fault_candidates = []
    for run in validation_runs:
        details = _run_diagnostics(run, model, apply_thresholds=False)
        if details["coverage"] < min_coverage:
            continue
        if run["label"] is None:
            eligible_healthy_anomalies.append(details["healthy_anomaly_score"])
            continue
        fault_candidates.append({
            "label": details["best_class"],
            "true_label": run["label"],
            "anomaly": details["healthy_anomaly_score"],
            "distance": details["best_distance"],
            "margin": details["class_margin"],
        })
    healthy_max = max(eligible_healthy_anomalies) if eligible_healthy_anomalies else None
    correct_faults = []
    for candidate in fault_candidates:
        if healthy_max is not None and candidate["anomaly"] <= healthy_max:
            continue
        best_class = candidate["label"]
        margin = candidate["margin"]
        if (best_class == candidate["true_label"] and
                (len(model["templates"]) == 1 or (margin is not None and margin > 1e-12))):
            correct_faults.append({
                "label": best_class,
                "distance": candidate["distance"],
                "margin": margin,
            })
    calibrated_labels = sorted({str(item["label"]) for item in correct_faults}, key=int)
    margins = [item["margin"] for item in correct_faults if item["margin"] is not None]
    return {
        "healthy_max_anomaly": healthy_max,
        "max_fault_distance": max((item["distance"] for item in correct_faults), default=None),
        "min_class_margin": min(margins, default=0.0),
        "calibrated_labels": calibrated_labels,
        "validation_runs_used": len(eligible_healthy_anomalies) + len(correct_faults),
        "validation_healthy_runs_used": len(eligible_healthy_anomalies),
        "validation_correct_fault_runs_used": len(correct_faults),
    }


def fit_dataset(
    dataset: Path | str,
    *,
    phases: str = "all",
    call_selection: str = "all",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve()
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be positive and finite")
    if phases not in {"all", "attention_backward"}:
        raise ValueError("phases must be all or attention_backward")
    if call_selection not in ALLOWED_CALL_SELECTIONS:
        raise ValueError(f"call_selection must be one of {sorted(ALLOWED_CALL_SELECTIONS)}")
    if call_selection == "first_in_phase" and phases != "attention_backward":
        raise ValueError("first_in_phase requires phases=attention_backward")
    catalog = _catalog_runs(dataset, load_labels_for={"train", "validation"})
    selected_catalog = [run for run in catalog if run["role"] in {"train", "validation"}]
    loaded = [
        _load_run(run, phases=phases, call_selection=call_selection)
        for run in selected_catalog
    ]
    train_runs = [run for run in loaded if run["role"] == "train"]
    validation_runs = [run for run in loaded if run["role"] == "validation"]
    if not train_runs:
        raise ValueError("fit requires train runs")
    if not validation_runs:
        raise ValueError("fit requires validation runs for refusal-threshold selection")
    invalid_train = [run for run in train_runs if run["feature_error"]]
    if invalid_train:
        details = {run["run_id"]: run["feature_error"] for run in invalid_train}
        raise ValueError(f"train runs cannot support first_in_phase: {details}")
    healthy_train = [run for run in train_runs if run["label"] is None]
    if len({run["session_id"] for run in healthy_train}) < 2:
        raise ValueError("healthy baselines need healthy train runs from at least two sessions")

    fault_labels = sorted({run["label"] for run in train_runs if run["label"] is not None})
    if not fault_labels:
        raise ValueError("train set has no fault_rank labels")
    templates: dict[str, list[dict[str, Any]]] = {str(label): [] for label in fault_labels}
    for label in fault_labels:
        class_runs = [run for run in train_runs if run["label"] == label]
        sessions = {run["session_id"] for run in class_runs}
        if len(sessions) < 2:
            raise ValueError(f"rank {label} needs fault train templates from at least two sessions")

    feature_ids = sorted({feature_id for run in healthy_train for feature_id in run["features"]})
    healthy_baselines: dict[str, dict[str, Any]] = {}
    for feature_id in feature_ids:
        values = [run["features"][feature_id] for run in healthy_train if feature_id in run["features"]]
        # Features must recur in independent healthy runs to receive a baseline.
        if len(values) < 2:
            continue
        center, scale = _robust_center_scale(values, scale_floor)
        healthy_baselines[feature_id] = {"center": center, "scale": scale, "n_train_runs": len(values)}
    feature_ids = sorted(healthy_baselines)
    if not feature_ids:
        raise ValueError("no stable features recur in at least two healthy train runs")

    for run in train_runs:
        if run["label"] is None:
            continue
        z_features = _normalized_features(run["features"], healthy_baselines)
        coverage = len(z_features) / len(feature_ids)
        if coverage < min_coverage:
            raise ValueError(
                f"fault template {run['run_id']} covers {coverage:.1%} of healthy features; "
                f"minimum is {min_coverage:.1%}"
            )
        templates[str(run["label"])].append({
            "run_id": run["run_id"],
            "session_id": run["session_id"],
            "values": z_features,
        })

    feature_definitions = []
    for feature_id in feature_ids:
        definition = json.loads(feature_id)
        definition["id"] = feature_id
        feature_definitions.append(definition)

    model: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "algorithm": "healthy_mad_normalized_nearest_run_template",
        "feature_spec": {
            "stable_topology_groups_only": True,
            "peer_center": "median(peer log duration_us) - rank log duration_us, per iteration",
            "run_aggregation": "median across iterations",
            "phase_code": "part of feature identity; absent legacy phase metadata maps to 0",
            "call_selection": call_selection,
            "call_selection_assumptions": (
                {
                    "scope": "Qwen TP2 PP2 with two microbatches; not a general call selector",
                    "group_label": "tensor_parallel",
                    "phase_subcode": 2,
                    "contiguous_blocks": FIRST_PHASE_BLOCK_COUNT,
                    "calls_per_block": FIRST_PHASE_BLOCK_SIZE,
                    "sampling_stride": FIRST_PHASE_SAMPLING_STRIDE,
                }
                if call_selection == "first_in_phase" else None
            ),
            "excluded_identifiers": ["clique", "group_id", "timestamp_us", "comm", "buffer_addr", "run_id", "fault_rank"],
        },
        "config": {
            "min_coverage": min_coverage,
            "scale_floor": scale_floor,
            "phases": phases,
            "call_selection": call_selection,
            "expected_phase_block_size": FIRST_PHASE_BLOCK_SIZE if call_selection == "first_in_phase" else None,
            "expected_blocks": FIRST_PHASE_BLOCK_COUNT if call_selection == "first_in_phase" else None,
            "sampling_stride": FIRST_PHASE_SAMPLING_STRIDE if call_selection == "first_in_phase" else None,
        },
        "feature_ids": feature_ids,
        "feature_definitions": feature_definitions,
        "healthy_baselines": healthy_baselines,
        "templates": templates,
        "class_labels": fault_labels,
        "training_counts": {
            "train_runs": len(train_runs),
            "healthy_train_runs": len(healthy_train),
            "fault_train_runs": sum(run["label"] is not None for run in train_runs),
            "train_sessions": sorted({run["session_id"] for run in train_runs}),
            "validation_runs": len(validation_runs),
            "validation_sessions": sorted({run["session_id"] for run in validation_runs}),
        },
        "source_dataset": str(dataset),
        "source_runs": _source_records(loaded),
        "held_out_sessions": sorted({run["session_id"] for run in catalog if run["role"] == "blind_test"}),
    }
    source_payload = [{"run_id": run["run_id"], "files": run["files"]} for run in model["source_runs"]]
    model["source_hash"] = _sha256_bytes(_canonical_json(source_payload).encode("utf-8"))
    model["thresholds"] = _calibrate_thresholds(validation_runs, model)
    model["frozen_model_hash"] = _sha256_bytes(_canonical_json(model).encode("utf-8"))
    return model


def verify_model(model: dict[str, Any]) -> None:
    if model.get("model_version") != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {model.get('model_version')}")
    frozen_hash = model.get("frozen_model_hash")
    payload = dict(model)
    payload.pop("frozen_model_hash", None)
    expected = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    if frozen_hash != expected:
        raise ValueError("frozen model hash mismatch; model file was changed")


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
    catalog = _catalog_runs(dataset, load_labels_for={split})
    train_sessions = set(model["training_counts"]["train_sessions"])
    validation_sessions = set(model["training_counts"]["validation_sessions"])
    selected = [run for run in catalog if run["role"] == split]
    if not selected:
        raise ValueError(f"dataset has no {split} runs")
    selected_sessions = {run["session_id"] for run in selected}
    forbidden_sources = (train_sessions | validation_sessions) if split == "blind_test" else train_sessions
    overlap = sorted(forbidden_sources & selected_sessions)
    if overlap:
        raise ValueError(f"{split} sessions overlap frozen fit sources: {overlap}")
    loaded = [
        _load_run(
            run,
            phases=model["config"]["phases"],
            call_selection=model["config"]["call_selection"],
        )
        for run in selected
    ]
    predictions = []
    for run in loaded:
        details = _run_diagnostics(run, model)
        true_label = "healthy" if run["label"] is None else f"rank:{run['label']}"
        predicted = details["predicted_label"]
        predicted_label = "healthy" if predicted is None else (
            "inconclusive" if predicted == "inconclusive" else f"rank:{predicted}"
        )
        predictions.append({
            **details,
            "true_label": true_label,
            "predicted": predicted_label,
        })
    known_labels = {"healthy", *(f"rank:{rank}" for rank in model["class_labels"])}
    true_labels = sorted(known_labels | {row["true_label"] for row in predictions},
                         key=lambda label: (label != "healthy", label))
    predicted_labels = sorted(known_labels | {"inconclusive"},
                              key=lambda label: (label != "healthy", label))
    matrix = {label: {prediction: 0 for prediction in predicted_labels} for label in true_labels}
    for row in predictions:
        matrix[row["true_label"]][row["predicted"]] += 1
    n_runs = len(predictions)
    n_correct = sum(row["true_label"] == row["predicted"] for row in predictions)
    inconclusive = sum(row["predicted"] == "inconclusive" for row in predictions)
    decided = n_runs - inconclusive
    healthy = [row for row in predictions if row["true_label"] == "healthy"]
    healthy_false_positives = sum(row["predicted"].startswith("rank:") for row in healthy)
    source_records = _source_records(loaded)
    source_payload = [{"run_id": run["run_id"], "files": run["files"]} for run in source_records]
    return {
        "report_version": 1,
        "split": split,
        "frozen_model_hash": model["frozen_model_hash"],
        "evaluation_source_hash": _sha256_bytes(_canonical_json(source_payload).encode("utf-8")),
        "evaluation_sources": source_records,
        "run_accuracy": n_correct / n_runs if n_runs else None,
        "correct_runs": n_correct,
        "run_denominator": n_runs,
        "decided_accuracy": n_correct / decided if decided else None,
        "decided_run_denominator": decided,
        "inconclusive_runs": inconclusive,
        "inconclusive_denominator": n_runs,
        "healthy_false_positives": healthy_false_positives,
        "healthy_false_positive_denominator": len(healthy),
        "confusion_matrix": matrix,
        "runs": predictions,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit healthy baselines and fault templates")
    fit.add_argument("--dataset", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--phases", choices=("all", "attention_backward"), default="all")
    fit.add_argument("--call-selection", choices=tuple(sorted(ALLOWED_CALL_SELECTIONS)), default="all")
    fit.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    fit.add_argument("--scale-floor", type=float, default=DEFAULT_SCALE_FLOOR)
    evaluate = commands.add_parser("evaluate", help="score runs with a frozen model")
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--split", required=True, choices=("validation", "blind_test"))
    evaluate.add_argument("--output", type=Path, help="write a JSON report instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fit":
            model = fit_dataset(
                args.dataset,
                phases=args.phases,
                call_selection=args.call_selection,
                min_coverage=args.min_coverage,
                scale_floor=args.scale_floor,
            )
            _write_json(args.output, model)
            print(json.dumps({
                "output": str(args.output),
                "frozen_model_hash": model["frozen_model_hash"],
                "source_hash": model["source_hash"],
                "features": len(model["feature_ids"]),
                "fault_classes": model["class_labels"],
                "thresholds": model["thresholds"],
            }, indent=2, ensure_ascii=False))
        else:
            model = _read_json(args.model)
            if not isinstance(model, dict):
                raise ValueError("model JSON must be an object")
            report = evaluate_dataset(args.dataset, model, split=args.split)
            if args.output:
                _write_json(args.output, report)
                print(json.dumps({
                    "output": str(args.output),
                    "split": args.split,
                    "frozen_model_hash": report["frozen_model_hash"],
                    "run_accuracy": report["run_accuracy"],
                    "correct_runs": report["correct_runs"],
                    "run_denominator": report["run_denominator"],
                    "inconclusive_runs": report["inconclusive_runs"],
                    "healthy_false_positives": report["healthy_false_positives"],
                }, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"offline: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
