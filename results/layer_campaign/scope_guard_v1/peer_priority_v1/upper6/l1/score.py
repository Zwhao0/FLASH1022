#!/usr/bin/env python3
"""Independent fixed-denominator scorer for continuation L1/L2 gates.

The two no-truth prediction files and execution-complete marker are required
before this scorer opens formal events, evidence or injection truth.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
SOURCE_FREEZE_PATH = HERE.parent / "source_freeze.json"
PARENT_SOURCE_FREEZE_PATH = HERE / "parent_guard_source_freeze.json"
PARENT_CORE_SOURCE_FREEZE_PATH = HERE / "parent_core_source_freeze.json"
ORIGINAL_STAGE_PATH = ROOT / "results/layer_campaign/scope_guard_v1/revalidation"
ORIGINAL_UPPER6_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/upper6"
PEER_MODEL_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped.json"
PEER_BINDING_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped_context_binding.json"
LAYOUTS = ("legacy", "node_local_pp")
RANKS = tuple(range(16))
ROLES = (("attention", "forward"), ("attention", "backward"),
         ("experts", "forward"), ("experts", "backward"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def strip_plan(value):
    return {key: item for key, item in value.items()
            if key not in ("session_id", "freeze_manifest_sha256", "scoped_reference_session")}


def prediction_path(out, layout):
    return out / f"{layout}_predictions_without_truth.json"


def recovery_bundle_checks(out, original_plan, source_freeze_path, design_path, ready):
    """Validate the explicit L2 PP infrastructure-recovery amendment, if present."""
    amendment_path = out / "recovery_amendment.json"
    recovery_path = out / "recovery_node_local_pp_plan.json"
    health_path = out / "recovery_health_plan.json"
    recovery_ready_path = out / "recovery_ready.json"
    paths = (amendment_path, recovery_path, health_path, recovery_ready_path)
    if not any(path.exists() for path in paths):
        return None, None
    if not all(path.is_file() for path in paths):
        return {"passed": False, "reason": "incomplete_recovery_bundle"}, None

    amendment = read_json(amendment_path)
    recovery = read_json(recovery_path)
    health = read_json(health_path)
    recovery_ready = read_json(recovery_ready_path)
    original_without_session = dict(original_plan)
    original_without_session.pop("session_id", None)
    recovery_without_session = dict(recovery)
    recovery_without_session.pop("session_id", None)

    artifact_checks = {}
    for name, digest in recovery_ready.get("artifacts", {}).items():
        path = SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name
        artifact_checks[name] = path.is_file() and sha(path) == digest

    bindings = amendment.get("bindings", {})
    semantic = amendment.get("semantic_equivalence", {})
    old_plan_hash = sha(out / "formal_node_local_pp_plan.json")
    recovery_hash = sha(recovery_path)
    health_hash = sha(health_path)
    source_hash = sha(source_freeze_path)
    design_hash = sha(design_path)
    infrastructure_path = out / "infrastructure_failure.json"
    checks = {
        "amendment_status": amendment.get("status") == "ready_for_owner_review",
        "recovery_ready_status": recovery_ready.get("status") == "recovery_ready_for_owner_review",
        "recovery_ready_artifact_hashes_match": bool(artifact_checks) and all(artifact_checks.values()),
        "original_plan_hash_matches": bindings.get("original_formal_plan_sha256") == old_plan_hash,
        "recovery_plan_hash_matches": bindings.get("recovery_formal_plan_sha256") == recovery_hash,
        "health_plan_hash_matches": bindings.get("recovery_health_plan_sha256") == health_hash,
        "source_freeze_hash_matches": bindings.get("source_freeze_sha256") == source_hash,
        "design_hash_matches": bindings.get("design_sha256") == design_hash,
        "original_ready_hash_matches": bindings.get("original_plan_ready_sha256") == sha(out / "plan_ready.json"),
        "timeonly_gate_hash_matches": bindings.get("timeonly_gate_sha256") == sha(out / "timeonly_gate.json"),
        "recovery_session_matches_amendment": recovery.get("session_id") == amendment.get("recovery_session"),
        "recovery_session_is_new": recovery.get("session_id") != original_plan.get("session_id"),
        "semantic_plan_equal_except_session_id": original_without_session == recovery_without_session,
        "semantic_hashes_match": (
            semantic.get("equal") is True
            and semantic.get("original_without_session_id_sha256") == canonical_sha256(original_without_session)
            and semantic.get("recovery_without_session_id_sha256") == canonical_sha256(recovery_without_session)
        ),
        "no_result_based_selection": amendment.get("no_result_based_selection") is True,
        "fixed_denominator_32": recovery.get("fixed_denominator") == 32,
        "recovery_steps_preserved": recovery.get("steps") == original_plan.get("steps") == 1049,
        "health_plan_42_steps": health.get("steps") == 42,
        "health_plan_opening_only": [(i.get("segment"), i.get("start"), i.get("end"))
                                      for i in health.get("intervals", [])] == [
                                          ("warmup", 1, 12), ("healthy", 13, 42)],
        "health_plan_inference_off": all(health.get(key) is False for key in
                                         ("online", "evidence_enabled", "probe_enabled", "profile_enabled")),
        "health_plan_no_injection": all(not i.get("target_ranks") and not i.get("fault_enabled")
                                        for i in health.get("intervals", [])),
        "infrastructure_sidecar_present": infrastructure_path.is_file(),
        "infrastructure_sidecar_hash_matches": bindings.get("infrastructure_failure_sha256") ==
                                                 (sha(infrastructure_path) if infrastructure_path.is_file() else None),
    }
    checks["passed"] = all(checks.values())
    return checks, recovery


def verify_bound_inputs(stage, layout, prediction, out):
    ready = read_json(out / "plan_ready.json")
    source_freeze_path = SOURCE_FREEZE_PATH
    source_freeze = read_json(source_freeze_path)
    plan_path = out / f"formal_{layout}_plan.json"
    parent_plan_path = out / f"parent_formal_{layout}_plan.json"
    design_path = out / "design.json"
    plan, parent_plan, design = read_json(plan_path), read_json(parent_plan_path), read_json(design_path)
    parent_dir = ORIGINAL_STAGE_PATH / stage
    parent_plan_actual_path = parent_dir / f"formal_{layout}_plan.json"
    parent_ready_path = parent_dir / "plan_ready.json"
    parent_amendment_path = parent_dir / "revalidation_amendment.json"
    parent_ready = read_json(parent_ready_path)
    parent_amendment = read_json(parent_amendment_path)
    original_plan_path = ORIGINAL_UPPER6_PATH / stage / f"formal_{layout}_plan.json"
    original_plan = read_json(original_plan_path)
    gate_path = out / "timeonly_gate.json"
    complete_path = out / "execution_complete.json"

    ready_hashes = {
        name: sha(source_freeze_path if name == "source_freeze.json" else out / name)
        for name in ready.get("artifacts", {})
    }
    source_ok = all((ROOT / name).is_file() and sha(ROOT / name) == digest
                    for name, digest in source_freeze.get("files", {}).items())
    remote = read_json(out / "remote_source_check.json")
    remote_ok = bool(remote) and all(
        item.get("passed") is True and item.get("files") == source_freeze.get("files")
        for item in remote.values()
    )

    # Verify the inherited revalidation bundle against its still-immutable source.
    parent_artifacts_ok = True
    for name, digest in parent_ready.get("artifacts", {}).items():
        source_path = PARENT_SOURCE_FREEZE_PATH if name == "source_freeze.json" else parent_dir / name
        parent_artifacts_ok = parent_artifacts_ok and source_path.is_file() and sha(source_path) == digest
    old_binding = parent_amendment.get("layouts", {}).get(layout, {})
    old_plan_hash_ok = (
        parent_amendment.get("stage") == stage.upper()
        and parent_amendment.get("parent_source_freeze_sha256") == sha(PARENT_CORE_SOURCE_FREEZE_PATH)
        and parent_amendment.get("source_freeze_sha256") == sha(PARENT_SOURCE_FREEZE_PATH)
        and parent_amendment.get("design_sha256") == sha(parent_dir / "design.json")
        and parent_amendment.get("timeonly_gate_sha256") == sha(parent_dir / "timeonly_gate.json")
        and old_binding.get("old_plan_sha256") == sha(original_plan_path)
        and old_binding.get("new_plan_sha256") == sha(parent_plan_actual_path)
        and old_binding.get("session_id") == parent_plan.get("session_id")
        and old_binding.get("catalog_session") == parent_plan.get("scoped_reference_session")
        and strip_plan(original_plan) == strip_plan(parent_plan)
    )
    exact_old_plan = read_json(parent_plan_actual_path) == parent_plan
    expected_plan = dict(parent_plan)
    expected_plan["session_id"] = f"v2s_peer_priority_upper6_{stage}_{layout}_01"
    expected_plan["freeze_manifest_sha256"] = sha(source_freeze_path)
    changed = sorted(k for k in set(expected_plan) | set(plan)
                     if expected_plan.get(k) != plan.get(k))

    live_session = prediction.get("session")
    session_path = ROOT / "results/data16_sessions" / str(live_session)
    live_plan_path = session_path / "host2/plan.json"
    live_plan = read_json(live_plan_path) if live_plan_path.is_file() else None
    complete = read_json(complete_path) if complete_path.is_file() else {}
    amendment = read_json(out / "upper6_rebinding_amendment.json")
    amendment_layout = amendment.get("layouts", {}).get(layout, {})
    gate = read_json(gate_path)

    checks = {
        "ready_status": ready.get("status") == "ready_for_single_gpu_owner_after_blind8_gate",
        "ready_artifacts_match": bool(ready_hashes) and ready_hashes == ready.get("artifacts"),
        "current_source_files_match": source_ok,
        "remote_source_matches": remote_ok,
        "parent_ready_artifacts_match": parent_artifacts_ok,
        "parent_revalidation_amendment_matches": old_plan_hash_ok and exact_old_plan,
        "parent_design_and_gate_preserved": (
            sha(out / "design.json") == sha(parent_dir / "design.json")
            and sha(out / "timeonly_gate.json") == sha(parent_dir / "timeonly_gate.json")
            and sha(out / "timeonly_ready.json") == sha(parent_dir / "timeonly_ready.json")
        ),
        "parent_source_binding": (
            amendment.get("parent_guard_freeze_sha256") == sha(PARENT_SOURCE_FREEZE_PATH)
            and amendment.get("parent_core_freeze_sha256") == sha(PARENT_CORE_SOURCE_FREEZE_PATH)
            and source_freeze.get("parent_source_freeze", {}).get("sha256") == sha(PARENT_SOURCE_FREEZE_PATH)
        ),
        "design_keeps_original_core_freeze": design.get("source_freeze", {}).get("sha256") == sha(PARENT_CORE_SOURCE_FREEZE_PATH),
        "plan_design_sha_matches": plan.get("design_sha256") == sha(design_path),
        "plan_freeze_sha_matches": plan.get("freeze_manifest_sha256") == sha(source_freeze_path),
        "plan_fields_only_session_and_freeze_changed": changed == ["freeze_manifest_sha256", "session_id"],
        "session_matches_prediction_and_amendment": (
            plan.get("session_id") == live_session == amendment_layout.get("session_id")
        ),
        "plan_matches_live_session": live_plan == plan,
        "complete_execution": complete.get("status") == "complete",
        "layout_matches": prediction.get("layout") == layout,
        "amendment_matches": (
            amendment.get("stage") == stage.upper()
            and amendment.get("source_freeze_sha256") == sha(source_freeze_path)
            and amendment.get("parent_guard_freeze_sha256") == sha(PARENT_SOURCE_FREEZE_PATH)
            and amendment.get("parent_revalidation_amendment_sha256") == sha(parent_amendment_path)
            and amendment.get("design_sha256") == sha(design_path)
            and amendment.get("timeonly_gate_sha256") == sha(gate_path)
            and amendment_layout.get("parent_plan_sha256") == sha(parent_plan_path)
            and amendment_layout.get("new_plan_sha256") == sha(plan_path)
            and amendment_layout.get("changed_fields") == ["freeze_manifest_sha256", "session_id"]
            and amendment_layout.get("preserved_fields_sha256") == canonical_sha256({
                k: v for k, v in plan.items() if k not in ("freeze_manifest_sha256", "session_id")
            })
        ),
        "inherited_timeonly_gate_stable": gate.get("stable_for_formal") is True,
    }
    checks["passed"] = all(checks.values())
    return checks, design, plan, session_path

def compact_timeline(events):
    timeline = {}
    for index, event in enumerate(events):
        epoch = event.get("epoch")
        if event.get("event") == "detection":
            timeline.setdefault(epoch, {}).setdefault("detections", []).append({
                "iteration": event.get("iteration"), "source": event.get("source"),
                "parent_epoch": event.get("parent_epoch"),
                "profile_iterations": event.get("profile_iterations"),
                "event_index": index,
            })
        elif event.get("event") == "profile_schedule":
            timeline.setdefault(epoch, {})["profile"] = {"start": event.get("start"), "end": event.get("end")}
        elif event.get("event") == "localization":
            result = event.get("result", {})
            timeline.setdefault(epoch, {})["localization"] = {
                "status": result.get("status"), "rank": result.get("rank"),
                "predicted": result.get("predicted"), "reason": result.get("reason"),
                "event_index": index,
            }
    return timeline


def evidence_check(schedule, all_evidence):
    from flash_v1.online_v2 import evidence_integrity_reason

    start, end, epoch = int(schedule["start"]), int(schedule["end"]), int(schedule["epoch"])
    selected = [r for r in all_evidence if r.get("epoch") == epoch and start <= r.get("iteration", -1) <= end]
    measurements = [r for r in selected if r.get("record_type") == "measurement"]
    markers = [r for r in selected if r.get("record_type") == "iteration_complete"]
    expected_pairs = {(iteration, rank) for iteration in range(start, end + 1) for rank in RANKS}
    marker_pairs = Counter((r.get("iteration"), r.get("rank")) for r in markers)
    expected_roles = {(iteration, rank, slot, phase) for iteration, rank in expected_pairs
                      for slot, phase in ROLES}
    role_counts = Counter((r.get("iteration"), r.get("rank"), r.get("slot"), r.get("phase"))
                          for r in measurements)
    markers_complete = len(markers) == 128 and set(marker_pairs) == expected_pairs and all(v == 1 for v in marker_pairs.values())
    roles_complete = len(measurements) == 512 and set(role_counts) == expected_roles and all(v == 1 for v in role_counts.values())
    marker_counts_valid = all(r.get("rows_count") == 4 and r.get("expected_rows") == 4 for r in markers)
    integrity_reason = evidence_integrity_reason(measurements, {"start": start, "end": end}, RANKS)
    return {
        "epoch": epoch, "start": start, "end": end,
        "measurement_count": len(measurements), "expected_measurements": 512,
        "completion_markers": len(markers), "expected_markers": 128,
        "roles_complete": roles_complete, "markers_complete": markers_complete,
        "marker_row_counts_valid": marker_counts_valid,
        "evidence_integrity_reason": integrity_reason,
        "passed": roles_complete and markers_complete and marker_counts_valid and integrity_reason is None,
    }


def auxiliary_check(schedule, all_auxiliary, session, layout, model, binding):
    """Offline audit of the peer channel, independent of the public dc89 result."""
    from flash_v1.stage_forward_localize import validate_row
    from flash_v1.stage_forward_peer import infer_bound, verify_topology

    if schedule is None or int(schedule["end"]) - int(schedule["start"]) + 1 != 8:
        return {"complete": False, "reason": "missing_or_non_eight_step_schedule"}
    start, end, epoch = int(schedule["start"]), int(schedule["end"]), int(schedule["epoch"])
    rows = [row for row in all_auxiliary if row.get("epoch") == epoch
            and start <= row.get("iteration", -1) <= end]
    expected = {(iteration, rank) for iteration in range(start, end + 1) for rank in RANKS}
    identities = [(row.get("iteration"), row.get("rank")) for row in rows]
    errors = []
    for row in rows:
        try:
            validate_row(row)
        except Exception as exc:
            errors.append(str(exc))
    topology_files = sorted(session.glob("host*/topology.json"))
    topology = None
    topology_ok = False
    if topology_files:
        try:
            topology = read_json(topology_files[0])
            topology_hash = canonical_sha256(topology)
            for path in topology_files:
                if canonical_sha256(read_json(path)) != topology_hash:
                    raise ValueError("host_topologies_differ")
            verify_topology(model, topology, layout)
            topology_ok = True
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"topology:{exc}")
    complete = (len(rows) == 128 and len(set(identities)) == 128
                and set(identities) == expected and not errors and topology_ok)
    peer = None
    if complete:
        peer = infer_bound(rows, model, RANKS, binding)
    pair_support = (peer or {}).get("pair_support", {})
    support = {str(pair): {"matched_iterations": int(detail.get("matched_iterations", 0)),
                           "unknown_iterations": int(detail.get("unknown_iterations", 0)),
                           "comparable": bool(detail.get("comparable", False))}
               for pair, detail in pair_support.items()}
    return {
        "schedule": {"epoch": epoch, "start": start, "end": end},
        "rows": len(rows), "expected_rows": 128,
        "unique_rank_iteration_pairs": len(set(identities)),
        "topology_valid": topology_ok, "validation_errors": errors[:8],
        "complete": complete,
        "status": (peer or {}).get("status", "not_inferred"),
        "rank": (peer or {}).get("rank"),
        "reason": (peer or {}).get("reason"),
        "support_by_pair": support,
        "known_pairs_support3": sum(v["matched_iterations"] >= 3 for v in support.values()),
        "result": peer,
    }


def core_snapshot(session, epoch):
    if epoch is None:
        return None
    path = session / "host2" / f"online_profile_{epoch}" / "original_dc89.json"
    return read_json(path) if path.is_file() else None


def verdict_bucket(result):
    if not isinstance(result, dict):
        return "missing"
    if result.get("status") == "localized" or str(result.get("predicted", "")).startswith("rank:"):
        return "localized"
    if result.get("status") == "rejected" or result.get("predicted") == "inconclusive":
        return "rejected"
    return "unknown"


def mean(values):
    return statistics.mean(values) if values else None


def extract_prediction(event):
    result = (event or {}).get("result", {})
    rank = result.get("rank")
    predicted = result.get("predicted", "inconclusive")
    if rank is None and isinstance(predicted, str) and predicted.startswith("rank:"):
        try:
            rank = int(predicted.split(":", 1)[1])
        except ValueError:
            pass
    return result, predicted, rank


def transition_audit(events, schedules, localizations, all_evidence):
    by_epoch_detection = {event.get("epoch"): event for event in events if event.get("event") == "detection"}
    by_epoch_localization = {event.get("epoch"): (index, event) for index, event in enumerate(events)
                             if event.get("event") == "localization"}
    transitions = [event for event in events if event.get("event") == "detection"
                   and event.get("source") == "profile_transition"]
    children = Counter(event.get("parent_epoch") for event in transitions)
    transition_rows = []
    violations = []
    for event in transitions:
        child, parent = event.get("epoch"), event.get("parent_epoch")
        parent_loc_pair = by_epoch_localization.get(parent)
        parent_profile = schedules.get(parent)
        child_profile = schedules.get(child)
        parent_evidence = evidence_check(parent_profile, all_evidence) if parent_profile else None
        child_evidence = evidence_check(child_profile, all_evidence) if child_profile else None
        parent_loc_index = parent_loc_pair[0] if parent_loc_pair else None
        event_index = events.index(event)
        checks = {
            "parent_epoch_present": parent is not None and parent in by_epoch_detection,
            "child_epoch_distinct": child is not None and child != parent,
            "parent_localization_before_transition": parent_loc_index is not None and parent_loc_index < event_index,
            "parent_profile_complete_evidence": bool(parent_evidence and parent_evidence["passed"]
                                                     and parent_profile["end"] - parent_profile["start"] + 1 == 8),
            "profile_iterations_is_8": event.get("profile_iterations") == 8,
            "child_profile_scheduled": bool(child_profile and child_profile["end"] - child_profile["start"] + 1 == 8),
            "parent_child_quota_one": children[parent] <= 1,
            "child_does_not_continue": children[child] == 0,
            "child_evidence_complete": bool(child_evidence and child_evidence["passed"]),
            "observed_iteration_present": type(event.get("iteration")) is int,
        }
        row = {
            "source": event.get("source"), "parent_epoch": parent, "child_epoch": child,
            "observed_iteration": event.get("iteration"), "profile_iterations": event.get("profile_iterations"),
            "parent_evidence": parent_evidence, "child_evidence": child_evidence,
            "checks": checks, "passed": all(checks.values()),
        }
        transition_rows.append(row)
        if not row["passed"]:
            violations.append({"parent_epoch": parent, "child_epoch": child,
                               "failed_checks": [k for k, value in checks.items() if not value]})
    return {
        "count": len(transitions), "passed": sum(r["passed"] for r in transition_rows),
        "violations": violations, "events": transition_rows,
    }


def score_layout(stage, layout, prediction, out):
    binding, design, plan, session = verify_bound_inputs(stage, layout, prediction, out)
    events = prediction.get("events", [])
    live_events = read_jsonl(session / "host2/events.jsonl")
    train = read_jsonl(session / "host2/train.jsonl")
    peer_model, peer_binding = read_json(PEER_MODEL_PATH), read_json(PEER_BINDING_PATH)
    from flash_v1.stage_forward_peer import verify_model
    verify_model(peer_model)
    if peer_binding.get("model_sha256") != peer_model.get("sha256"):
        raise ValueError("peer model/context binding mismatch")
    if plan.get("stage_forward_aux_enabled") is not True:
        raise ValueError("formal plan did not enable the frozen peer auxiliary")
    if (plan.get("stage_forward_aux_model_file_sha256") != sha(PEER_MODEL_PATH)
            or plan.get("stage_forward_aux_binding_file_sha256") != sha(PEER_BINDING_PATH)):
        raise ValueError("formal plan peer model/binding hashes do not match")
    public_timeline, live_timeline = compact_timeline(events), compact_timeline(live_events)
    all_epochs = set(public_timeline) | set(live_timeline)
    event_match = {"epochs": len(all_epochs),
                   "matching": sum(public_timeline.get(e) == live_timeline.get(e) for e in all_epochs),
                   "mismatching_epochs": [e for e in sorted(all_epochs)
                                          if public_timeline.get(e) != live_timeline.get(e)]}

    detections = [event for event in events if event.get("event") == "detection"]
    detections = sorted((e for e in detections if type(e.get("iteration")) is int), key=lambda e: e["iteration"])
    schedules = {event.get("epoch"): event for event in events if event.get("event") == "profile_schedule"}
    localizations = {event.get("epoch"): event for event in events if event.get("event") == "localization"}
    cases = {case["case_id"]: case for case in design["layouts"][layout]}
    faults = [interval for interval in plan["intervals"] if interval.get("segment") == "fault"]
    expected_per_layout = 1 if stage == "l1" else 2
    if len(cases) != expected_per_layout or len(faults) != expected_per_layout:
        raise ValueError(f"fixed denominator mismatch for {stage}/{layout}")
    if stage == "l2":
        phase_counts = Counter(case.get("phase") for case in cases.values())
        if phase_counts != Counter({"fwd": 1, "bwd": 1}) or len({case["rank"] for case in cases.values()}) != 2:
            raise ValueError(f"L2 must contain one FWD and one BWD case on distinct ranks: {layout}")

    all_evidence = []
    for path in sorted(session.glob("host*/evidence_rank*.jsonl")):
        all_evidence.extend(read_jsonl(path))
    all_auxiliary = []
    for path in sorted(session.glob("host*/stage_forward_aux_rank*.jsonl")):
        all_auxiliary.extend(read_jsonl(path))
    auxiliary_by_epoch = {
        int(epoch): auxiliary_check(schedule, all_auxiliary, session, layout, peer_model, peer_binding)
        for epoch, schedule in schedules.items()
    }
    truth_by_rank = {}
    for host in ("host1", "host2"):
        for path in (session / host).glob("whole_compute_truth_rank*.jsonl"):
            rank = int(path.stem.rsplit("rank", 1)[1])
            truth_by_rank[rank] = read_jsonl(path)
    train_by_iter = {int(row["iteration"]): row for row in train}

    rows = []
    for fault in faults:
        case_id, start, end = fault["case_id"], int(fault["start"]), int(fault["end"])
        case = cases[case_id]
        target = int(case["rank"])
        fault_detections = [event for event in detections if start <= int(event["iteration"]) <= end]
        primary = fault_detections[0] if fault_detections else None
        epoch = primary.get("epoch") if primary else None
        loc_event = localizations.get(epoch) if epoch is not None else None
        loc_result, predicted, predicted_rank = extract_prediction(loc_event)
        detected = primary is not None
        rank_correct = bool(detected and loc_result.get("status") == "localized" and predicted_rank == target)
        outcome = ("not_triggered" if not detected else
                   "rejected" if loc_result.get("status") != "localized" or predicted_rank is None else
                   "correct_rank" if rank_correct else "wrong_rank")

        profile = schedules.get(epoch) if epoch is not None else None
        profile_start = int(profile["start"]) if profile else None
        profile_end = int(profile["end"]) if profile else None
        profile_complete = bool(profile and profile_end - profile_start + 1 == 8)
        profile_inside = bool(profile_complete and start <= profile_start and profile_end <= end)
        profile_integrity = evidence_check(profile, all_evidence) if profile else None
        peer_audit = auxiliary_by_epoch.get(int(epoch)) if epoch is not None else None
        core_result = core_snapshot(session, epoch)
        core_status = verdict_bucket(core_result)
        core_rank = core_result.get("rank") if isinstance(core_result, dict) else None
        if core_rank is None and isinstance(core_result, dict):
            core_rank = extract_prediction({"result": core_result})[2]
        peer_outcome = ("missing" if not peer_audit else
                        "invalid" if not peer_audit.get("complete") else
                        "rejected" if peer_audit.get("status") != "localized" or peer_audit.get("rank") is None else
                        "correct_rank" if int(peer_audit["rank"]) == target else "wrong_rank")

        truth = [row for row in truth_by_rank.get(target, []) if start <= int(row.get("iteration", -1)) <= end]
        expected_steps = end - start + 1
        selected_calls = sum(int(row.get("selected_compute_calls", 0) or 0) for row in truth)
        launches = sum(int(row.get("injected_launches", 0) or 0) for row in truth)
        misses = sum(row.get("selected_scope_missed") is True for row in truth)
        drift = sum(bool(row.get("scope_trace_drift")) for row in truth)
        injection_valid = bool(len(truth) == expected_steps and selected_calls > 0 and launches > 0
                               and misses == 0 and drift == 0
                               and all(row.get("scope_execution_valid") is True for row in truth)
                               and all(row.get("case_id") == case_id for row in truth))

        previous_healthy = next((interval for interval in plan["intervals"]
                                 if interval.get("id") == f"healthy_after_{case_id}"), None)
        if previous_healthy is None:
            previous_healthy = next((interval for interval in plan["intervals"]
                                     if interval.get("id") == f"healthy_opening"), None)
        healthy_times = [float(train_by_iter[i]["wall_iteration_ms"])
                         for i in range(previous_healthy["start"], previous_healthy["end"] + 1)
                         if previous_healthy and i in train_by_iter
                         and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0]
        fault_times = [float(train_by_iter[i]["wall_iteration_ms"]) for i in range(start, end + 1)
                       if i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0]
        healthy_mean, fault_mean = mean(healthy_times), mean(fault_times)
        nonprofile_degradation = 100.0 * (fault_mean / healthy_mean - 1.0) if healthy_mean and fault_mean is not None else None

        strict = bool(rank_correct and profile_complete and profile_inside and profile_integrity
                      and profile_integrity["passed"] and injection_valid)
        rows.append({
            "case_id": case_id, "layer": stage.upper(), "layout": layout,
            "category": case["category"], "phase": case.get("phase"), "rank": target,
            "fault_start": start, "fault_end": end,
            "detected_in_fault": detected, "detection_source": primary.get("source") if primary else None,
            "detection_iteration": primary.get("iteration") if primary else None,
            "detection_latency_steps": int(primary["iteration"]) - start if primary else None,
            "detection_epoch": epoch, "detection_parent_epoch": primary.get("parent_epoch") if primary else None,
            "fault_window_detection_count": len(fault_detections),
            "fault_window_detection_sources": [e.get("source", "beast") for e in fault_detections],
            "outcome": outcome, "predicted": predicted, "predicted_rank": predicted_rank,
            "public_localization_source": loc_result.get("source", "dc89_core"),
            "core_status": core_status, "core_rank": core_rank,
            "core_outcome": ("missing" if core_result is None else
                             "rejected" if core_status != "localized" else
                             "correct_rank" if core_rank == target else "wrong_rank"),
            "online_peer_status": (loc_result.get("stage_forward_aux") or {}).get("status"),
            "online_peer_reason": (loc_result.get("stage_forward_aux") or {}).get("reason"),
            "peer_offline_outcome": peer_outcome,
            "peer_offline_status": peer_audit.get("status") if peer_audit else None,
            "peer_offline_rank": peer_audit.get("rank") if peer_audit else None,
            "peer_offline_known_pairs_support3": peer_audit.get("known_pairs_support3") if peer_audit else None,
            "peer_offline_complete": peer_audit.get("complete") if peer_audit else False,
            "peer_offline_reason": peer_audit.get("reason") if peer_audit else None,
            "localization_reason": loc_result.get("reason"),
            "profile_start": profile_start, "profile_end": profile_end,
            "profile_complete_8_steps": profile_complete,
            "profile_entirely_within_fault": profile_inside,
            "profile_evidence_integrity": profile_integrity,
            "selected_compute_calls": selected_calls, "physical_fma_launches": launches,
            "selected_miss_steps": misses, "scope_trace_drift_steps": drift,
            "injection_valid": injection_valid,
            "healthy_nonprofile_steps": len(healthy_times), "fault_nonprofile_steps": len(fault_times),
            "healthy_mean_ms": healthy_mean, "fault_mean_ms": fault_mean,
            "fault_nonprofile_wall_degradation_pct": nonprofile_degradation,
            "end_to_end_success": strict,
        })

    fault_intervals = [range(int(f["start"]), int(f["end"]) + 1) for f in faults]
    fault_steps = {step for values in fault_intervals for step in values}
    healthy_detections = [event for event in detections if int(event["iteration"]) > 12
                          and int(event["iteration"]) not in fault_steps]
    healthy_verdicts = []
    for event in healthy_detections:
        epoch = event.get("epoch")
        loc = localizations.get(epoch)
        result = (loc or {}).get("result", {})
        bucket = verdict_bucket(result)
        healthy_verdicts.append({"epoch": epoch, "iteration": event.get("iteration"),
                                 "source": result.get("source", "dc89_core") if isinstance(result, dict) else None,
                                 "verdict": bucket,
                                 "rank": extract_prediction(loc)[2] if loc else None,
                                 "peer_status": ((result.get("stage_forward_aux") or {}).get("status")
                                                 if isinstance(result, dict) else None),
                                 "peer_offline": auxiliary_by_epoch.get(int(epoch)) if epoch is not None else None})
    transition = transition_audit(events, schedules, localizations, all_evidence)
    expected_health_steps = {int(row["iteration"]) for row in train
                             if int(row.get("iteration", 0)) > 12 and int(row["iteration"]) not in fault_steps}
    healthy_nonprofile = sum(i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0
                             for i in expected_health_steps)
    return {
        "layout": layout, "session": prediction.get("session"),
        "prediction_file": str(prediction_path(out, layout).relative_to(ROOT)),
        "prediction_sha256": sha(prediction_path(out, layout)),
        "binding": binding, "prediction_vs_online": event_match,
        "cases": rows,
        "healthy_false_triggers": len(healthy_detections),
        "healthy_false_trigger_sources": dict(Counter(e.get("source", "beast") for e in healthy_detections)),
        "healthy_trigger_verdicts": dict(Counter(row["verdict"] for row in healthy_verdicts)),
        "healthy_localized_false_alarms": sum(row["verdict"] == "localized" for row in healthy_verdicts),
        "healthy_trigger_localizations": healthy_verdicts,
        "healthy_nonfault_nonwarmup_steps": len(expected_health_steps),
        "healthy_nonfault_nonwarmup_nonprofile_steps": healthy_nonprofile,
        "peer_offline_window_audits": [auxiliary_by_epoch[epoch] for epoch in sorted(auxiliary_by_epoch)],
        "profile_transition_audit": transition,
    }


def summarize(stage, layouts):
    cases = [case for result in layouts.values() for case in result["cases"]]
    denominator = 2 if stage == "l1" else 4
    counts = Counter(case["outcome"] for case in cases)
    public_sources = Counter(case["public_localization_source"] for case in cases)
    peer_outcomes = Counter(case["peer_offline_outcome"] for case in cases)
    core_outcomes = Counter(case["core_outcome"] for case in cases)
    per_layout = {}
    for layout, result in layouts.items():
        rows = result["cases"]
        per_layout[layout] = {
            "denominator": 1 if stage == "l1" else 2,
            "strict_success": sum(case["end_to_end_success"] for case in rows),
            "detected": sum(case["detected_in_fault"] for case in rows),
            "correct_rank": sum(case["outcome"] == "correct_rank" for case in rows),
            "wrong_rank": sum(case["outcome"] == "wrong_rank" for case in rows),
            "rejected": sum(case["outcome"] == "rejected" for case in rows),
            "not_triggered": sum(case["outcome"] == "not_triggered" for case in rows),
            "healthy_false_triggers": result["healthy_false_triggers"],
            "healthy_false_trigger_sources": result["healthy_false_trigger_sources"],
            "healthy_trigger_verdicts": result["healthy_trigger_verdicts"],
            "healthy_localized_false_alarms": result["healthy_localized_false_alarms"],
            "healthy_nonfault_nonwarmup_steps": result["healthy_nonfault_nonwarmup_steps"],
            "healthy_nonfault_nonwarmup_nonprofile_steps": result["healthy_nonfault_nonwarmup_nonprofile_steps"],
            "valid_injections": sum(case["injection_valid"] for case in rows),
            "full_profiles_inside_fault": sum(case["profile_complete_8_steps"] and case["profile_entirely_within_fault"] for case in rows),
            "prediction_vs_online": result["prediction_vs_online"],
            "profile_transition_audit": result["profile_transition_audit"],
            "peer_offline_windows": len(result["peer_offline_window_audits"]),
            "peer_offline_complete_windows": sum(w.get("complete", False) for w in result["peer_offline_window_audits"]),
        }
    if stage == "l2":
        by_phase = {}
        for phase in ("fwd", "bwd"):
            selected = [case for case in cases if case.get("phase") == phase]
            by_phase[phase] = {"denominator": 2,
                               "strict_success": sum(case["end_to_end_success"] for case in selected),
                               "detected": sum(case["detected_in_fault"] for case in selected),
                               "wrong_rank": sum(case["outcome"] == "wrong_rank" for case in selected),
                               "rejected": sum(case["outcome"] == "rejected" for case in selected),
                               "not_triggered": sum(case["outcome"] == "not_triggered" for case in selected)}
    else:
        by_phase = None
    return {
        "protocol": f"upper6_{stage}_score_v1",
        "status": "complete", "fixed_denominator": denominator,
        "strict_success": sum(case["end_to_end_success"] for case in cases),
        "strict_success_pct": 100.0 * sum(case["end_to_end_success"] for case in cases) / denominator,
        "detected": sum(case["detected_in_fault"] for case in cases),
        "correct_rank": counts["correct_rank"], "wrong_rank": counts["wrong_rank"],
        "rejected": counts["rejected"], "not_triggered": counts["not_triggered"],
        "public_localization_sources": dict(public_sources),
        "core_outcomes": dict(core_outcomes), "peer_offline_outcomes": dict(peer_outcomes),
        "peer_offline_complete_fault_windows": sum(case["peer_offline_complete"] for case in cases),
        "peer_offline_localized_correct": sum(case["peer_offline_outcome"] == "correct_rank" for case in cases),
        "peer_offline_localized_wrong": sum(case["peer_offline_outcome"] == "wrong_rank" for case in cases),
        "peer_offline_rejected_or_invalid": sum(case["peer_offline_outcome"] in ("rejected", "invalid") for case in cases),
        "injection_valid": sum(case["injection_valid"] for case in cases),
        "complete_profiles_inside_fault": sum(case["profile_complete_8_steps"] and case["profile_entirely_within_fault"] for case in cases),
        "healthy_false_triggers": sum(result["healthy_false_triggers"] for result in layouts.values()),
        "by_layout": per_layout, "by_phase": by_phase,
        "profile_transition_events": sum(result["profile_transition_audit"]["count"] for result in layouts.values()),
        "profile_transition_passed": sum(result["profile_transition_audit"]["passed"] for result in layouts.values()),
        "prediction_first": True,
        "gate_passed": sum(case["end_to_end_success"] for case in cases) == denominator,
        "gate_rule": "L1 requires 2/2; L2 requires 4/4. All cases stay in the fixed denominator.",
        "cases": cases,
    }


def run(stage=None):
    stage = stage or HERE.name
    out = HERE
    expected = 2 if stage == "l1" else 4
    # Read both blind artifacts before opening the training, event, evidence, or truth files.
    paths = {layout: prediction_path(out, layout) for layout in LAYOUTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"blind predictions must precede truth scoring: {missing}")
    predictions = {layout: read_json(path) for layout, path in paths.items()}
    if any(predictions[layout].get("layout") != layout for layout in LAYOUTS):
        raise ValueError("prediction layout mismatch")
    complete = read_json(out / "execution_complete.json")
    if complete.get("status") != "complete":
        raise RuntimeError("formal execution is not complete")

    layouts = {layout: score_layout(stage, layout, predictions[layout], out) for layout in LAYOUTS}
    result = summarize(stage, layouts)
    result["layout_results"] = layouts
    if len(result["cases"]) != expected:
        result["status"] = "partial"
    atomic_json(out / "cases.json", {"protocol": result["protocol"], "fixed_denominator": expected,
                                     "prediction_first": True, "layouts": layouts})
    atomic_json(out / "summary.json", result)
    return result


def main():
    stage = sys.argv[1] if len(sys.argv) == 2 else HERE.name
    if stage not in ("l1", "l2") or len(sys.argv) > 2:
        raise SystemExit("usage: score.py [l1|l2]")
    result = run(stage)
    print(json.dumps({"summary": str(HERE / "summary.json"),
                      "stage": stage, "fixed_denominator": result["fixed_denominator"],
                      "strict_success": result["strict_success"],
                      "detected": result["detected"], "wrong_rank": result["wrong_rank"],
                      "rejected": result["rejected"], "not_triggered": result["not_triggered"],
                      "healthy_false_triggers": result["healthy_false_triggers"],
                      "gate_passed": result["gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
