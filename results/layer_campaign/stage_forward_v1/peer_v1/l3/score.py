#!/usr/bin/env python3
"""Independent CPU scorer for the fixed 32-case L3 component acceptance run.

Both no-truth prediction snapshots and the completion marker are required.
The scorer freezes and checks predictions/events before it opens training,
evidence, auxiliary sidecars, or injection truth.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
PEER_DIR = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_FREEZE_PATH = PEER_DIR / "source_freeze.json"
CORE_MODEL_PATH = ROOT / "results/v2_semantic/linear_candidate_model.json"
PEER_MODEL_PATH = PEER_DIR / "model_scoped.json"
PEER_BINDING_PATH = PEER_DIR / "model_scoped_context_binding.json"
UPPER_SCORE_PATH = PEER_DIR / "upper6/score.py"
LAYOUTS = ("legacy", "node_local_pp")
RANKS = tuple(range(16))
FAMILIES = ("attention", "routed_expert_mlp", "moe_block_router", "normalization")
ROLES = (("attention", "forward"), ("attention", "backward"),
         ("experts", "forward"), ("experts", "backward"))
ALLOWED_EVENTS = {"detection", "profile_schedule", "localization"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


upper = _load("peer_v1_upper6_scorer_for_l3", UPPER_SCORE_PATH)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def prediction_path(layout):
    return HERE / f"{layout}_predictions_without_truth.json"


def interval_faults(plan):
    return [row for row in plan.get("intervals", []) if row.get("segment") == "fault"]


def verify_design_plan(layout, design, plan, source_freeze, source_sha):
    cases = design.get("layouts", {}).get(layout, [])
    faults = interval_faults(plan)
    if len(cases) != 16 or len(faults) != 16:
        raise ValueError(f"L3 {layout} must preserve 16 design and fault cases")
    if len({row.get("rank") for row in cases}) != 16 or {int(row["rank"]) for row in cases} != set(RANKS):
        raise ValueError(f"L3 {layout} must target every rank once")
    family_counts = Counter(row.get("family") for row in cases)
    if family_counts != Counter({family: 4 for family in FAMILIES}):
        raise ValueError(f"L3 {layout} family quota mismatch: {dict(family_counts)}")
    if (plan.get("steps") != 1049 or plan.get("fixed_denominator") != 32
            or plan.get("layout") != layout or plan.get("parallel_layout") != layout
            or plan.get("seed") != design.get("training_seed")):
        raise ValueError(f"L3 {layout} plan identity or fixed 1049-step schedule mismatch")
    if (plan.get("online") is not True or plan.get("evidence_enabled") is not True
            or plan.get("profile_enabled") is not False or plan.get("probe_enabled") is not False
            or plan.get("profile_backend") != "semantic_only"
            or plan.get("scoped_compute_enabled") is not True
            or plan.get("fault_kind") != "component_compute"
            or plan.get("component_operator_population_phase") != "fwd"
            or plan.get("component_family_quotas_per_layout") != {family: 4 for family in FAMILIES}
            or plan.get("monitor_profile_transition") is not True
            or plan.get("stage_forward_aux_enabled") is not True
            or plan.get("stage_forward_aux_model") != str(PEER_MODEL_PATH.relative_to(ROOT))
            or plan.get("stage_forward_aux_context_binding") != str(PEER_BINDING_PATH.relative_to(ROOT))):
        raise ValueError(f"L3 {layout} online/core/peer plan flags mismatch")
    core = read(CORE_MODEL_PATH)
    if (source_freeze.get("model_sha256") != core.get("sha256")
            or plan.get("frozen_model_sha256") != core.get("sha256")
            or plan.get("design_sha256") != sha(HERE / "design.json")
            or plan.get("freeze_manifest_sha256") != source_sha):
        raise ValueError(f"L3 {layout} core/source freeze binding mismatch")
    if (plan.get("stage_forward_aux_model_file_sha256") != sha(PEER_MODEL_PATH)
            or plan.get("stage_forward_aux_binding_file_sha256") != sha(PEER_BINDING_PATH)):
        raise ValueError(f"L3 {layout} peer model/context binding file hash mismatch")
    by_id = {row.get("case_id"): row for row in cases}
    if len(by_id) != 16:
        raise ValueError(f"L3 {layout} case ids must be unique")
    if {row.get("case_id") for row in faults} != set(by_id):
        raise ValueError(f"L3 {layout} fault intervals do not match design cases")
    for fault in faults:
        case = by_id[fault["case_id"]]
        rank = int(case["rank"])
        scope = {"kind": "component", "phase": "fwd", "slot_id": case["site"]["slot_id"]}
        if (fault.get("rank") != rank or fault.get("target_ranks") != [rank]
                or fault.get("fault_mode") != "whole_compute" or fault.get("fault_scope") != scope
                or fault.get("phase") != "fwd"
                or fault.get("slowdown_extra_ratio") != case["dose"]["slowdown_extra_ratio"]
                or int(fault["end"]) - int(fault["start"]) + 1 != 32):
            raise ValueError(f"L3 {layout} case scope/phase/dose mismatch: {fault['case_id']}")
    return {"case_count": 16, "family_counts": dict(sorted(family_counts.items())),
            "rank_population": 16, "formal_steps": 1049}


def verify_pretruth_bundle():
    ready_path = HERE / "plan_ready.json"
    timeonly_gate_path = HERE / "timeonly_gate.json"
    design_path = HERE / "design.json"
    if not all(path.is_file() for path in (ready_path, timeonly_gate_path, design_path)):
        raise FileNotFoundError("L3 ready, design and time-only gate artifacts must exist")
    ready, gate, design = read(ready_path), read(timeonly_gate_path), read(design_path)
    if (ready.get("status") != "ready_for_single_gpu_owner" or ready.get("stage") != "L3"
            or ready.get("fixed_denominator") != 32 or gate.get("stable_for_formal") is not True
            or design.get("integration_not_blind") is not False
            or design.get("independent_randomized_acceptance") is not True
            or design.get("prediction_before_truth") is not True
            or design.get("family_quotas_per_layout") != {family: 4 for family in FAMILIES}
            or design.get("phase") != "fwd"):
        raise ValueError("L3 execution contract does not describe the frozen independent 32-case acceptance run")
    freeze = read(SOURCE_FREEZE_PATH)
    source_sha = sha(SOURCE_FREEZE_PATH)
    if (ready.get("source_freeze_sha256") != source_sha
            or gate.get("source_freeze_sha256") != source_sha
            or design.get("source_freeze", {}).get("sha256") != source_sha):
        raise ValueError("L3 bundle source-freeze hashes disagree")
    runtime_mismatches = {name: {"expected": digest, "actual": sha(ROOT / name) if (ROOT / name).is_file() else None}
                          for name, digest in freeze.get("files", {}).items()
                          if not (ROOT / name).is_file() or sha(ROOT / name) != digest}
    if runtime_mismatches:
        raise ValueError(f"L3 runtime no longer matches frozen source: {runtime_mismatches}")
    model, binding = read(PEER_MODEL_PATH), read(PEER_BINDING_PATH)
    from flash_v1.stage_forward_peer import verify_model
    verify_model(model)
    if (binding.get("model_sha256") != model.get("sha256")
            or binding.get("sha256") != canonical_sha({k: v for k, v in binding.items() if k != "sha256"})):
        raise ValueError("L3 peer model binding is invalid")
    artifact_checks = {}
    for name, digest in ready.get("artifacts", {}).items():
        path = SOURCE_FREEZE_PATH if name == "source_freeze.json" else HERE / name
        artifact_checks[name] = path.is_file() and sha(path) == digest
    if not artifact_checks or not all(artifact_checks.values()):
        raise ValueError(f"L3 ready artifact hash mismatch: {artifact_checks}")

    plans, sessions = {}, {}
    for layout in LAYOUTS:
        plan_path = HERE / f"formal_{layout}_plan.json"
        plan = read(plan_path)
        plan_checks = verify_design_plan(layout, design, plan, freeze, source_sha)
        session_name = f"v2s_peer_l3_formal_{layout}_01"
        if plan.get("session_id") != session_name:
            raise ValueError(f"unexpected L3 session id for {layout}")
        session = ROOT / "results/data16_sessions" / session_name
        live_plan_path = session / "plan.json"
        if not live_plan_path.is_file() or read(live_plan_path) != plan:
            raise ValueError(f"L3 {layout} session plan does not match the frozen plan")
        prediction_file = prediction_path(layout)
        if not prediction_file.is_file():
            raise FileNotFoundError(f"no-truth prediction snapshot is missing: {prediction_file}")
        prediction = read(prediction_file)
        if (prediction.get("layout") != layout or prediction.get("session") != session_name
                or any(event.get("event") not in ALLOWED_EVENTS for event in prediction.get("events", []))):
            raise ValueError(f"invalid L3 no-truth prediction snapshot: {layout}")
        events_path = session / "host2/events.jsonl"
        actual_events = [row for row in read_jsonl(events_path) if row.get("event") in ALLOWED_EVENTS]
        if prediction.get("events") != actual_events:
            raise ValueError(f"prediction snapshot differs from live controller events: {layout}")
        sessions[layout] = {"session": session, "prediction": prediction,
                            "prediction_path": prediction_file, "events_path": events_path,
                            "plan": plan, "plan_checks": plan_checks,
                            "prediction_sha256": sha(prediction_file),
                            "events_sha256": sha(events_path),
                            "events_canonical_sha256": canonical_sha(actual_events)}
        plans[layout] = {"plan_sha256": sha(plan_path), **plan_checks}
    complete_path = HERE / "execution_complete.json"
    if not complete_path.is_file() or read(complete_path).get("status") != "complete":
        raise RuntimeError("both L3 layouts must be complete before truth scoring")
    complete = read(complete_path)
    if complete.get("fixed_denominator") != 32:
        raise ValueError("L3 completion marker does not retain fixed denominator 32")
    return {"ready": ready, "gate": gate, "design": design, "freeze": freeze,
            "peer_model": model, "peer_binding": binding, "plans": plans,
            "sessions": sessions, "completion_sha256": sha(complete_path)}


def persist_prediction_first(bundle):
    """Persist an immutable hash map after checking events, before loading any truth."""
    path = HERE / "scorer_prediction_freeze.json"
    record = {"protocol": "peer_v1_l3_prediction_freeze_v1", "prediction_before_truth": True,
              "execution_complete_sha256": bundle["completion_sha256"], "layouts": {}}
    for layout, data in bundle["sessions"].items():
        record["layouts"][layout] = {"prediction_path": str(data["prediction_path"].relative_to(ROOT)),
            "prediction_sha256": data["prediction_sha256"],
            "events_path": str(data["events_path"].relative_to(ROOT)),
            "events_sha256": data["events_sha256"],
            "events_canonical_sha256": data["events_canonical_sha256"],
            "plan_sha256": sha(HERE / f"formal_{layout}_plan.json"),
            "session": data["session"].name}
    if path.exists():
        if read(path) != record:
            raise FileExistsError("preserve existing prediction freeze; it differs from current inputs")
    else:
        atomic_json(path, record)
    return record


def mean(values):
    return statistics.mean(values) if values else None


def result_for(event):
    result = (event or {}).get("result", {})
    predicted, rank = result.get("predicted", "inconclusive"), result.get("rank")
    if rank is None and isinstance(predicted, str) and predicted.startswith("rank:"):
        try:
            rank = int(predicted.split(":", 1)[1])
        except ValueError:
            pass
    return result, predicted, rank


def load_case_truth(session, rank):
    for host in ("host1", "host2"):
        path = session / host / f"whole_compute_truth_rank{rank}.jsonl"
        rows = read_jsonl(path)
        if rows:
            return rows, path
    return [], None


def score_layout(layout, data, bundle):
    session, plan, design, prediction = (data["session"], data["plan"],
                                        bundle["design"], data["prediction"])
    events = prediction["events"]
    train = read_jsonl(session / "host2/train.jsonl")
    train_by_iter = {int(row["iteration"]): row for row in train if type(row.get("iteration")) is int}
    if len(train_by_iter) != 1049 or set(train_by_iter) != set(range(1, 1050)):
        raise ValueError(f"L3 {layout} train iteration grid incomplete")
    schedules = {int(event["epoch"]): event for event in events if event.get("event") == "profile_schedule"}
    localizations = {int(event["epoch"]): event for event in events if event.get("event") == "localization"}
    detections = sorted((event for event in events if event.get("event") == "detection"
                         and type(event.get("iteration")) is int), key=lambda event: event["iteration"])
    all_evidence = []
    for path in sorted(session.glob("host*/evidence_rank*.jsonl")):
        all_evidence.extend(read_jsonl(path))
    all_auxiliary = []
    for path in sorted(session.glob("host*/stage_forward_aux_rank*.jsonl")):
        all_auxiliary.extend(read_jsonl(path))
    auxiliary_by_epoch = {
        epoch: upper.auxiliary_check(schedule, all_auxiliary, session, layout,
                                     bundle["peer_model"], bundle["peer_binding"])
        for epoch, schedule in schedules.items()}
    faults = interval_faults(plan)
    cases_by_id = {row["case_id"]: row for row in design["layouts"][layout]}
    case_rows = []
    all_fault_steps = set()
    for fault in faults:
        case = cases_by_id[fault["case_id"]]
        start, end, target = int(fault["start"]), int(fault["end"]), int(case["rank"])
        all_fault_steps.update(range(start, end + 1))
        in_fault = [event for event in detections if start <= int(event["iteration"]) <= end]
        primary = in_fault[0] if in_fault else None
        epoch = int(primary["epoch"]) if primary and type(primary.get("epoch")) is int else None
        loc_event = localizations.get(epoch) if epoch is not None else None
        result, predicted, predicted_rank = result_for(loc_event)
        detected = primary is not None
        localized = result.get("status") == "localized" or str(predicted).startswith("rank:")
        if not detected:
            outcome = "not_triggered"
        elif not localized or predicted_rank is None:
            outcome = "rejected"
        elif int(predicted_rank) == target:
            outcome = "correct_rank"
        else:
            outcome = "wrong_rank"

        profile = schedules.get(epoch) if epoch is not None else None
        profile_start = int(profile["start"]) if profile else None
        profile_end = int(profile["end"]) if profile else None
        profile_complete = bool(profile and profile_end - profile_start + 1 == 8)
        profile_inside = bool(profile_complete and start <= profile_start and profile_end <= end)
        evidence_integrity = upper.evidence_check(profile, all_evidence) if profile else None
        peer = auxiliary_by_epoch.get(epoch) if epoch is not None else None
        core = upper.core_snapshot(session, epoch)
        _, _, core_rank = result_for({"result": core}) if core else ({}, "inconclusive", None)
        core_bucket = upper.verdict_bucket(core)
        if core is None:
            core_outcome = "missing"
        elif core_bucket != "localized" or core_rank is None:
            core_outcome = "rejected"
        else:
            core_outcome = "correct_rank" if int(core_rank) == target else "wrong_rank"
        peer_bucket = ("missing" if not peer else "invalid" if not peer.get("complete") else
                       "rejected" if peer.get("status") != "localized" or peer.get("rank") is None else
                       "correct_rank" if int(peer["rank"]) == target else "wrong_rank")
        source = result.get("source", "dc89_core") if isinstance(result, dict) else "dc89_core"

        truth, truth_path = load_case_truth(session, target)
        truth = [row for row in truth if start <= int(row.get("iteration", -1)) <= end]
        case_id = case["case_id"]
        selected_calls = sum(int(row.get("selected_compute_calls", 0) or 0) for row in truth)
        launches = sum(int(row.get("injected_launches", 0) or 0) for row in truth)
        misses = sum(row.get("selected_scope_missed") is True for row in truth)
        drift = sum(bool(row.get("scope_trace_drift")) for row in truth)
        valid_steps = sum(row.get("scope_execution_valid") is True for row in truth)
        truth_cases = Counter(row.get("case_id") for row in truth)
        injection_valid = (len(truth) == 32 and selected_calls > 0 and launches > 0 and misses == 0
                           and drift == 0 and valid_steps == 32 and truth_cases == Counter({case_id: 32}))

        preceding = [interval for interval in plan["intervals"]
                     if interval.get("segment") == "healthy" and int(interval["end"]) < start]
        baseline_interval = max(preceding, key=lambda row: int(row["end"])) if preceding else None
        healthy_wall = ([float(train_by_iter[i]["wall_iteration_ms"])
                         for i in range(int(baseline_interval["start"]), int(baseline_interval["end"]) + 1)
                         if i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0]
                        if baseline_interval else [])
        fault_wall = [float(train_by_iter[i]["wall_iteration_ms"]) for i in range(start, end + 1)
                      if i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0]
        healthy_mean, fault_mean = mean(healthy_wall), mean(fault_wall)
        degradation = 100 * (fault_mean / healthy_mean - 1) if healthy_mean and fault_mean is not None else None
        strict = bool(outcome == "correct_rank" and profile_complete and profile_inside
                      and evidence_integrity and evidence_integrity["passed"] and injection_valid)
        case_rows.append({
            "case_id": case_id, "layout": layout, "family": case["family"], "category": case["category"],
            "rank": target, "slot_id": case["site"]["slot_id"], "module_path": case["site"].get("path"),
            "fault_start": start, "fault_end": end, "detection_count_in_fault": len(in_fault),
            "detected_in_fault": detected,
            "detection_iteration": primary.get("iteration") if primary else None,
            "detection_latency_steps": int(primary["iteration"]) - start if primary else None,
            "detection_source": primary.get("source") if primary else None,
            "detection_epoch": epoch, "outcome": outcome, "public_localization_source": source,
            "public_predicted": predicted, "public_rank": predicted_rank,
            "public_reason": result.get("reason"), "core_outcome": core_outcome,
            "core_rank": core_rank,
            "online_peer_status": (result.get("stage_forward_aux") or {}).get("status"),
            "online_peer_reason": (result.get("stage_forward_aux") or {}).get("reason"),
            "peer_offline_outcome": peer_bucket, "peer_offline_status": peer.get("status") if peer else None,
            "peer_offline_rank": peer.get("rank") if peer else None,
            "peer_offline_reason": peer.get("reason") if peer else None,
            "peer_offline_complete": bool(peer and peer.get("complete")),
            "peer_offline_known_pairs_support3": peer.get("known_pairs_support3") if peer else None,
            "peer_offline_support_by_pair": peer.get("support_by_pair") if peer else None,
            "profile_start": profile_start, "profile_end": profile_end,
            "profile_start_latency_steps": profile_start - start if profile_start is not None else None,
            "profile_end_latency_steps": profile_end - start if profile_end is not None else None,
            "profile_complete_8_steps": profile_complete, "profile_entirely_within_fault": profile_inside,
            "profile_evidence_integrity": evidence_integrity,
            "selected_compute_calls": selected_calls, "physical_fma_launches": launches,
            "selected_miss_steps": misses, "scope_trace_drift_steps": drift,
            "scope_execution_valid_steps": valid_steps, "truth_rows": len(truth),
            "truth_path": str(truth_path.relative_to(ROOT)) if truth_path else None,
            "injection_valid": injection_valid,
            "preceding_healthy_interval": baseline_interval,
            "healthy_nonprofile_steps": len(healthy_wall), "fault_nonprofile_steps": len(fault_wall),
            "healthy_mean_ms": healthy_mean, "fault_mean_ms": fault_mean,
            "fault_nonprofile_wall_degradation_pct": degradation,
            "end_to_end_success": strict,
        })

    healthy_detections = [event for event in detections if int(event["iteration"]) > 12
                          and int(event["iteration"]) not in all_fault_steps]
    healthy_verdicts = []
    for event in healthy_detections:
        epoch = int(event["epoch"]) if type(event.get("epoch")) is int else None
        localization = localizations.get(epoch) if epoch is not None else None
        result, predicted, predicted_rank = result_for(localization)
        healthy_verdicts.append({"iteration": event.get("iteration"), "epoch": epoch,
            "source": result.get("source", "dc89_core") if isinstance(result, dict) else None,
            "verdict": upper.verdict_bucket(result), "rank": predicted_rank,
            "reason": result.get("reason") if isinstance(result, dict) else None,
            "peer_offline": auxiliary_by_epoch.get(epoch) if epoch is not None else None})
    transition = upper.transition_audit(events, schedules, localizations, all_evidence)
    expected_health = {int(row["iteration"]) for row in train
                       if int(row.get("iteration", 0)) > 12 and int(row["iteration"]) not in all_fault_steps}
    healthy_nonprofile = sum(i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0
                             for i in expected_health)
    return {"layout": layout, "session": data["session"].name,
            "prediction_sha256": data["prediction_sha256"],
            "plan_sha256": sha(HERE / f"formal_{layout}_plan.json"),
            "plan_checks": data["plan_checks"], "cases": case_rows,
            "healthy_false_triggers": len(healthy_detections),
            "healthy_false_trigger_verdicts": dict(Counter(row["verdict"] for row in healthy_verdicts)),
            "healthy_wrong_gpu_localizations": sum(row["verdict"] == "localized" for row in healthy_verdicts),
            "healthy_trigger_localizations": healthy_verdicts,
            "healthy_nonfault_nonwarmup_steps": len(expected_health),
            "healthy_nonfault_nonwarmup_nonprofile_steps": healthy_nonprofile,
            "peer_offline_window_audits": [auxiliary_by_epoch[epoch] for epoch in sorted(auxiliary_by_epoch)],
            "profile_transition_audit": transition}


def latency_stats(cases, key):
    vals = [int(row[key]) for row in cases if row.get(key) is not None]
    return {"n": len(vals), "missing": len(cases) - len(vals),
            "min": min(vals) if vals else None, "median": statistics.median(vals) if vals else None,
            "max": max(vals) if vals else None, "values": vals}


def summarize(layout_results):
    cases = [case for layout in LAYOUTS for case in layout_results[layout]["cases"]]
    outcome = Counter(case["outcome"] for case in cases)
    family_results = {}
    for family in FAMILIES:
        selected = [case for case in cases if case["family"] == family]
        family_results[family] = {"fixed_denominator": 8,
            "strict_success": sum(row["end_to_end_success"] for row in selected),
            "detected": sum(row["detected_in_fault"] for row in selected),
            "correct_rank": sum(row["outcome"] == "correct_rank" for row in selected),
            "wrong_rank": sum(row["outcome"] == "wrong_rank" for row in selected),
            "rejected": sum(row["outcome"] == "rejected" for row in selected),
            "not_triggered": sum(row["outcome"] == "not_triggered" for row in selected),
            "valid_fma_cases": sum(row["injection_valid"] for row in selected),
            "peer_offline_outcomes": dict(Counter(row["peer_offline_outcome"] for row in selected))}
    layout_summary = {}
    for layout, result in layout_results.items():
        rows = result["cases"]
        layout_summary[layout] = {"fixed_denominator": 16,
            "strict_success": sum(row["end_to_end_success"] for row in rows),
            "detected": sum(row["detected_in_fault"] for row in rows),
            "correct_rank": sum(row["outcome"] == "correct_rank" for row in rows),
            "wrong_rank": sum(row["outcome"] == "wrong_rank" for row in rows),
            "rejected": sum(row["outcome"] == "rejected" for row in rows),
            "not_triggered": sum(row["outcome"] == "not_triggered" for row in rows),
            "healthy_false_triggers": result["healthy_false_triggers"],
            "healthy_false_trigger_verdicts": result["healthy_false_trigger_verdicts"],
            "healthy_wrong_gpu_localizations": result["healthy_wrong_gpu_localizations"],
            "healthy_steps": result["healthy_nonfault_nonwarmup_steps"],
            "healthy_nonprofile_steps": result["healthy_nonfault_nonwarmup_nonprofile_steps"],
            "public_sources": dict(Counter(row["public_localization_source"] for row in rows)),
            "core_outcomes": dict(Counter(row["core_outcome"] for row in rows)),
            "peer_offline_outcomes": dict(Counter(row["peer_offline_outcome"] for row in rows)),
            "peer_complete_windows": sum(row["peer_offline_complete"] for row in rows),
            "peer_support3_pairs": sum(row["peer_offline_known_pairs_support3"] or 0 for row in rows),
            "valid_fma_cases": sum(row["injection_valid"] for row in rows),
            "profile_transition_count": result["profile_transition_audit"]["count"],
            "profile_transition_valid": result["profile_transition_audit"]["passed"]}
    strict = sum(row["end_to_end_success"] for row in cases)
    denom = 32
    return {"protocol": "peer_v1_l3_score_v1", "status": "complete", "fixed_denominator": denom,
        "strict_success": strict, "strict_success_pct": 100.0 * strict / denom,
        "gate_threshold": 29, "gate_passed": strict >= 29,
        "gate_rule": "at least 29/32 strict cases; failed, weak, rejected and invalid cases remain in denominator",
        "detected": sum(row["detected_in_fault"] for row in cases),
        "outcomes": dict(outcome), "public_sources": dict(Counter(row["public_localization_source"] for row in cases)),
        "correct_public_sources": dict(Counter(
            row["public_localization_source"] for row in cases if row["outcome"] == "correct_rank")),
        "core_outcomes": dict(Counter(row["core_outcome"] for row in cases)),
        "peer_offline_outcomes": dict(Counter(row["peer_offline_outcome"] for row in cases)),
        "peer_offline_complete_fault_windows": sum(row["peer_offline_complete"] for row in cases),
        "peer_offline_localized_correct": sum(row["peer_offline_outcome"] == "correct_rank" for row in cases),
        "peer_offline_localized_wrong": sum(row["peer_offline_outcome"] == "wrong_rank" for row in cases),
        "peer_offline_known_pairs_support3_total": sum(row["peer_offline_known_pairs_support3"] or 0 for row in cases),
        "healthy_false_triggers": sum(result["healthy_false_triggers"] for result in layout_results.values()),
        "healthy_wrong_gpu_localizations": sum(result["healthy_wrong_gpu_localizations"] for result in layout_results.values()),
        "valid_fma_cases": sum(row["injection_valid"] for row in cases),
        "profile_complete_inside_fault": sum(row["profile_complete_8_steps"] and row["profile_entirely_within_fault"] for row in cases),
        "latency_steps": {"detection": latency_stats(cases, "detection_latency_steps"),
                          "profile_start": latency_stats(cases, "profile_start_latency_steps"),
                          "profile_end": latency_stats(cases, "profile_end_latency_steps")},
        "profile_transition_events": sum(result["profile_transition_audit"]["count"] for result in layout_results.values()),
        "profile_transition_valid": sum(result["profile_transition_audit"]["passed"] for result in layout_results.values()),
        "by_layout": layout_summary, "by_family": family_results,
        "prediction_before_truth": True,
        "cases": cases}


def run():
    # This phase reads only plans, predictions, event logs and source hashes.
    # No train/evidence/auxiliary/truth file is opened until the freeze exists.
    bundle = verify_pretruth_bundle()
    prediction_freeze = persist_prediction_first(bundle)
    layout_results = {layout: score_layout(layout, bundle["sessions"][layout], bundle)
                      for layout in LAYOUTS}
    summary = summarize(layout_results)
    summary.update({"prediction_freeze": str((HERE / "scorer_prediction_freeze.json").relative_to(ROOT)),
                    "prediction_freeze_sha256": sha(HERE / "scorer_prediction_freeze.json"),
                    "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
                    "peer_model_sha256": bundle["peer_model"].get("sha256"),
                    "core_model_sha256": bundle["freeze"].get("model_sha256"),
                    "prediction_inputs": prediction_freeze,
                    "layout_results": layout_results})
    atomic_json(HERE / "cases.json", {"protocol": summary["protocol"],
                "fixed_denominator": 32, "prediction_before_truth": True,
                "layouts": layout_results})
    atomic_json(HERE / "summary.json", summary)
    return summary


def main():
    result = run()
    print(json.dumps({"summary": str(HERE / "summary.json"),
        "fixed_denominator": result["fixed_denominator"], "strict_success": result["strict_success"],
        "gate_passed": result["gate_passed"], "outcomes": result["outcomes"],
        "healthy_false_triggers": result["healthy_false_triggers"],
        "valid_fma_cases": result["valid_fma_cases"],
        "peer_offline_complete_fault_windows": result["peer_offline_complete_fault_windows"]}, indent=2))


if __name__ == "__main__":
    main()
