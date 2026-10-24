#!/usr/bin/env python3
"""Independent fixed-eight scorer for the component blind8 campaign.

The scorer requires completed sessions and pre-frozen public predictions.  It
persists both prediction snapshots before reading training evidence or fault
truth, and retains every case in the denominator.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
PEER_DIR = HERE.parent
SOURCE_FREEZE_PATH = PEER_DIR / "source_freeze.json"
CORE_MODEL_PATH = ROOT / "results/v2_semantic/linear_candidate_model.json"
PEER_MODEL_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped.json"
PEER_BINDING_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped_context_binding.json"
UPPER_SCORE_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/upper6/score.py"
LAYOUTS = ("legacy", "node_local_pp")
RANKS = tuple(range(16))
FAMILIES = ("attention", "routed_expert_mlp", "moe_block_router", "normalization")
ROLES = (("attention", "forward"), ("attention", "backward"),
         ("experts", "forward"), ("experts", "backward"))
ALLOWED_EVENTS = {"detection", "profile_schedule", "localization"}
FIXED_DENOMINATOR = 8
CASES_PER_LAYOUT = 4
STEPS_PER_LAYOUT = 305


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


upper = _load("blind8_upper6_score", UPPER_SCORE_PATH)
prepare = _load("blind8_prepare_for_score", HERE / "prepare.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def prediction_path(layout):
    return HERE / f"{layout}_predictions_without_truth.json"


def interval_faults(plan):
    return [row for row in plan.get("intervals", []) if row.get("segment") == "fault"]


def verify_design_plan(layout, design, plan, freeze, source_sha):
    cases = design.get("layouts", {}).get(layout, [])
    faults = interval_faults(plan)
    if len(cases) != CASES_PER_LAYOUT or len(faults) != CASES_PER_LAYOUT:
        raise ValueError(f"blind8 {layout} must preserve four design/fault cases")
    families = Counter(row.get("family") for row in cases)
    if families != Counter({family: 1 for family in FAMILIES}):
        raise ValueError(f"blind8 {layout} family quotas mismatch: {dict(families)}")
    ranks = [int(row["rank"]) for row in cases]
    if len(set(ranks)) != CASES_PER_LAYOUT or any(rank not in RANKS for rank in ranks):
        raise ValueError(f"blind8 {layout} must retain four distinct valid ranks")
    if (plan.get("steps") != STEPS_PER_LAYOUT or plan.get("fixed_denominator") != FIXED_DENOMINATOR
            or plan.get("layout") != layout or plan.get("parallel_layout") != layout
            or plan.get("seed") != design.get("training_seed")):
        raise ValueError(f"blind8 {layout} identity/denominator/schedule mismatch")
    if (plan.get("online") is not True or plan.get("evidence_enabled") is not True
            or plan.get("profile_enabled") is not False or plan.get("probe_enabled") is not False
            or plan.get("profile_backend") != "semantic_only"
            or plan.get("scoped_compute_enabled") is not True
            or plan.get("fault_kind") != "component_compute"
            or plan.get("component_operator_population_phase") != "fwd"
            or plan.get("component_family_quotas_per_layout") != {family: 1 for family in FAMILIES}
            or plan.get("monitor_profile_transition") is not True
            or plan.get("stage_forward_aux_enabled") is not True
            or plan.get("stage_forward_aux_model") != str(PEER_MODEL_PATH.relative_to(ROOT))
            or plan.get("stage_forward_aux_context_binding") != str(PEER_BINDING_PATH.relative_to(ROOT))):
        raise ValueError(f"blind8 {layout} core/peer/source plan flags mismatch")
    core = read(CORE_MODEL_PATH)
    if (freeze.get("model_sha256") != core.get("sha256")
            or plan.get("frozen_model_sha256") != core.get("sha256")
            or plan.get("design_sha256") != sha(HERE / "design.json")
            or plan.get("freeze_manifest_sha256") != source_sha
            or plan.get("scoped_reference_session") != f"v2s_scope_guard_catalog_{layout}_01"
            or plan.get("whole_compute_reference_session") != f"v2s_scope_catalog_{layout}_02"):
        raise ValueError(f"blind8 {layout} model/source/catalog reference mismatch")
    if (plan.get("stage_forward_aux_model_file_sha256") != sha(PEER_MODEL_PATH)
            or plan.get("stage_forward_aux_binding_file_sha256") != sha(PEER_BINDING_PATH)):
        raise ValueError("blind8 scoped peer model/binding hash mismatch")
    by_id = {row.get("case_id"): row for row in cases}
    if len(by_id) != CASES_PER_LAYOUT or {row.get("case_id") for row in faults} != set(by_id):
        raise ValueError(f"blind8 {layout} fault intervals do not match fixed cases")
    for fault in faults:
        case = by_id[fault["case_id"]]
        target, scope = int(case["rank"]), case["fault_target"]
        if (fault.get("rank") != target or fault.get("target_ranks") != [target]
                or fault.get("fault_mode") != "whole_compute" or fault.get("fault_scope") != scope
                or scope != {"kind": "component", "phase": "fwd", "slot_id": case["site"]["slot_id"]}
                or fault.get("phase") != "fwd"
                or fault.get("slowdown_extra_ratio") != case["dose"]["slowdown_extra_ratio"]
                or int(fault["end"]) - int(fault["start"]) + 1 != 32):
            raise ValueError(f"blind8 {layout} site/phase/dose/fault interval mismatch: {fault['case_id']}")
    return {"case_count": CASES_PER_LAYOUT, "family_counts": dict(sorted(families.items())),
            "distinct_target_ranks": sorted(ranks), "formal_steps": STEPS_PER_LAYOUT}


def verify_pretruth_bundle():
    ready_path = HERE / "plan_ready.json"
    design_lock_path = HERE / "design_lock.json"
    gate_path = HERE / "timeonly_gate.json"
    completion_path = HERE / "execution_complete.json"
    required = (ready_path, design_lock_path, gate_path, completion_path, HERE / "design.json")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("blind8 ready, design, gate and completion artifacts must exist")
    ready, lock, gate, complete, design = (read(ready_path), read(design_lock_path),
                                           read(gate_path), read(completion_path), read(HERE / "design.json"))
    if (ready.get("status") != "ready_for_single_gpu_owner_after_layer_gates"
            or ready.get("fixed_denominator") != FIXED_DENOMINATOR
            or gate.get("stable_for_formal") is not True
            or gate.get("timeonly_ready_sha256") != sha(HERE / "timeonly_ready.json")
            or complete.get("status") != "complete"
            or complete.get("fixed_denominator") != FIXED_DENOMINATOR
            or lock.get("gpu_authorized") is not False
            or design.get("integration_not_blind") is not False
            or design.get("independent_randomized_acceptance") is not True
            or design.get("prediction_before_truth") is not True
            or design.get("fixed_denominator") != FIXED_DENOMINATOR
            or design.get("cases_per_layout") != CASES_PER_LAYOUT
            or design.get("family_quotas_per_layout") != {family: 1 for family in FAMILIES}
            or design.get("component_operator_population_phase") != "fwd_only"):
        raise ValueError("blind8 execution contract is incomplete or denominator changed")
    source_sha = sha(SOURCE_FREEZE_PATH)
    freeze = read(SOURCE_FREEZE_PATH)
    if (ready.get("source_freeze_sha256") != source_sha
            or gate.get("source_freeze_sha256") != source_sha
            or design.get("source_freeze", {}).get("sha256") != source_sha):
        raise ValueError("blind8 source-freeze hashes disagree")
    runtime_mismatches = [name for name, digest in freeze.get("files", {}).items()
                          if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if runtime_mismatches:
        raise ValueError(f"blind8 runtime differs from frozen source: {runtime_mismatches[:8]}")
    from flash_v1.stage_forward_peer import verify_model
    peer_model, peer_binding = read(PEER_MODEL_PATH), read(PEER_BINDING_PATH)
    verify_model(peer_model)
    if peer_binding.get("model_sha256") != peer_model.get("sha256"):
        raise ValueError("blind8 peer model/context binding is invalid")

    artifact_checks = {}
    for name, digest in ready.get("artifacts", {}).items():
        path = SOURCE_FREEZE_PATH if name == "source_freeze.json" else HERE / name
        artifact_checks[name] = path.is_file() and sha(path) == digest
    if not artifact_checks or not all(artifact_checks.values()):
        raise ValueError(f"blind8 formal-ready artifact hash mismatch: {artifact_checks}")

    sessions = {}
    plans = {}
    for layout in LAYOUTS:
        plan_path = HERE / f"formal_{layout}_plan.json"
        plan = read(plan_path)
        checks = verify_design_plan(layout, design, plan, freeze, source_sha)
        session_name = f"v2s_peer_blind8_formal_{layout}_01"
        if plan.get("session_id") != session_name:
            raise ValueError(f"unexpected blind8 {layout} formal session")
        session = ROOT / "results/data16_sessions" / session_name
        live_plan = session / "host2/plan.json"
        if not live_plan.is_file() or read(live_plan) != plan:
            raise ValueError(f"blind8 {layout} live session plan differs from the frozen formal plan")
        pred_path = prediction_path(layout)
        if not pred_path.is_file():
            raise FileNotFoundError(f"blind8 no-truth prediction snapshot missing: {pred_path}")
        prediction = read(pred_path)
        if (prediction.get("layout") != layout or prediction.get("session") != session_name
                or any(event.get("event") not in ALLOWED_EVENTS for event in prediction.get("events", []))):
            raise ValueError(f"invalid blind8 public prediction snapshot: {layout}")
        events_path = session / "host2/events.jsonl"
        live_events = [row for row in read_jsonl(events_path) if row.get("event") in ALLOWED_EVENTS]
        if prediction.get("events") != live_events:
            raise ValueError(f"blind8 prediction snapshot differs from live public events: {layout}")
        sessions[layout] = {"session": session, "prediction": prediction,
                            "prediction_path": pred_path, "events_path": events_path,
                            "plan": plan, "plan_checks": checks,
                            "prediction_sha256": sha(pred_path), "events_sha256": sha(events_path),
                            "events_canonical_sha256": canonical_sha(live_events)}
        plans[layout] = {"plan_sha256": sha(plan_path), **checks}
    return {"ready": ready, "gate": gate, "design": design, "freeze": freeze,
            "peer_model": peer_model, "peer_binding": peer_binding,
            "sessions": sessions, "plans": plans,
            "completion_sha256": sha(completion_path)}


def persist_prediction_first(bundle):
    path = HERE / "scorer_prediction_freeze.json"
    record = {"protocol": "peer_priority_blind8_prediction_freeze_v1",
              "prediction_before_truth": True,
              "execution_complete_sha256": bundle["completion_sha256"], "layouts": {}}
    for layout, data in bundle["sessions"].items():
        record["layouts"][layout] = {
            "prediction_path": str(data["prediction_path"].relative_to(ROOT)),
            "prediction_sha256": data["prediction_sha256"],
            "events_path": str(data["events_path"].relative_to(ROOT)),
            "events_sha256": data["events_sha256"],
            "events_canonical_sha256": data["events_canonical_sha256"],
            "plan_sha256": sha(HERE / f"formal_{layout}_plan.json"),
            "session": data["session"].name,
        }
    if path.exists():
        if read(path) != record:
            raise FileExistsError("preserve the existing prediction freeze; current public predictions differ")
    else:
        atomic_json(path, record)
    return record


def result_for(event):
    result = (event or {}).get("result", {})
    predicted, rank = result.get("predicted", "inconclusive"), result.get("rank")
    if rank is None and isinstance(predicted, str) and predicted.startswith("rank:"):
        try:
            rank = int(predicted.split(":", 1)[1])
        except ValueError:
            rank = None
    return result, predicted, rank


def load_truth(session, rank):
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
    if len(train_by_iter) != STEPS_PER_LAYOUT or set(train_by_iter) != set(range(1, STEPS_PER_LAYOUT + 1)):
        raise ValueError(f"blind8 {layout} training iteration grid is incomplete")
    schedules = {int(event["epoch"]): event for event in events
                 if event.get("event") == "profile_schedule" and type(event.get("epoch")) is int}
    localizations = {int(event["epoch"]): event for event in events
                     if event.get("event") == "localization" and type(event.get("epoch")) is int}
    detections = sorted((event for event in events if event.get("event") == "detection"
                         and type(event.get("iteration")) is int), key=lambda event: event["iteration"])
    all_evidence, all_auxiliary = [], []
    for path in sorted(session.glob("host*/evidence_rank*.jsonl")):
        all_evidence.extend(read_jsonl(path))
    for path in sorted(session.glob("host*/stage_forward_aux_rank*.jsonl")):
        all_auxiliary.extend(read_jsonl(path))
    peer_model, peer_binding = bundle["peer_model"], bundle["peer_binding"]
    auxiliary = {epoch: upper.auxiliary_check(schedule, all_auxiliary, session, layout,
                                              peer_model, peer_binding)
                 for epoch, schedule in schedules.items()}
    faults = interval_faults(plan)
    cases_by_id = {row["case_id"]: row for row in design["layouts"][layout]}
    case_rows, fault_steps = [], set()
    for fault in faults:
        case = cases_by_id[fault["case_id"]]
        start, end, target = int(fault["start"]), int(fault["end"]), int(case["rank"])
        fault_steps.update(range(start, end + 1))
        in_fault = [event for event in detections if start <= int(event["iteration"]) <= end]
        primary = in_fault[0] if in_fault else None
        epoch = int(primary["epoch"]) if primary and type(primary.get("epoch")) is int else None
        location = localizations.get(epoch) if epoch is not None else None
        result, predicted, predicted_rank = result_for(location)
        if not primary:
            outcome = "not_triggered"
        elif result.get("status") != "localized" or predicted_rank is None:
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
        evidence = upper.evidence_check(profile, all_evidence) if profile else None
        peer = auxiliary.get(epoch) if epoch is not None else None
        core = upper.core_snapshot(session, epoch)
        core_result, _, core_rank = result_for({"result": core}) if core else ({}, "inconclusive", None)
        core_bucket = upper.verdict_bucket(core)
        if core is None:
            core_outcome = "missing"
        elif core_bucket != "localized" or core_rank is None:
            core_outcome = "rejected"
        else:
            core_outcome = "correct_rank" if int(core_rank) == target else "wrong_rank"
        peer_outcome = ("missing" if not peer else "invalid" if not peer.get("complete") else
                        "rejected" if peer.get("status") != "localized" or peer.get("rank") is None else
                        "correct_rank" if int(peer["rank"]) == target else "wrong_rank")
        public_source = result.get("source", "dc89_core") if isinstance(result, dict) else "dc89_core"

        truth, truth_path = load_truth(session, target)
        truth = [row for row in truth if start <= int(row.get("iteration", -1)) <= end]
        calls = sum(int(row.get("selected_compute_calls", 0) or 0) for row in truth)
        launches = sum(int(row.get("injected_launches", 0) or 0) for row in truth)
        misses = sum(row.get("selected_scope_missed") is True for row in truth)
        drift = sum(bool(row.get("scope_trace_drift")) for row in truth)
        valid_steps = sum(row.get("scope_execution_valid") is True for row in truth)
        case_id_counts = Counter(row.get("case_id") for row in truth)
        injection_valid = (len(truth) == 32 and calls > 0 and launches > 0 and misses == 0
                           and drift == 0 and valid_steps == 32
                           and case_id_counts == Counter({case["case_id"]: 32}))

        preceding = [interval for interval in plan["intervals"]
                     if interval.get("segment") == "healthy" and int(interval["end"]) < start]
        baseline_interval = max(preceding, key=lambda row: int(row["end"])) if preceding else None
        healthy_wall = ([float(train_by_iter[i]["wall_iteration_ms"])
                         for i in range(int(baseline_interval["start"]), int(baseline_interval["end"]) + 1)
                         if i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0]
                        if baseline_interval else [])
        fault_wall = [float(train_by_iter[i]["wall_iteration_ms"])
                      for i in range(start, end + 1)
                      if i in train_by_iter and int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0]
        healthy_median = statistics.median(healthy_wall) if healthy_wall else None
        fault_median = statistics.median(fault_wall) if fault_wall else None
        degradation = (100.0 * (fault_median / healthy_median - 1.0)
                       if healthy_median and fault_median is not None else None)
        strict = bool(outcome == "correct_rank" and profile_complete and profile_inside
                      and evidence and evidence["passed"] and injection_valid)
        case_rows.append({
            "case_id": case["case_id"], "layout": layout,
            "family": case["family"], "rank": target, "phase": "fwd",
            "slot_id": case["site"]["slot_id"], "module_path": case["site"].get("path"),
            "fault_start": start, "fault_end": end,
            "detection_count_in_fault": len(in_fault), "detected_in_fault": primary is not None,
            "detection_iteration": primary.get("iteration") if primary else None,
            "detection_latency_steps": int(primary["iteration"]) - start if primary else None,
            "detection_source": primary.get("source") if primary else None,
            "detection_epoch": epoch, "outcome": outcome,
            "public_localization_source": public_source,
            "public_predicted": predicted, "public_rank": predicted_rank,
            "public_reason": result.get("reason"), "core_outcome": core_outcome, "core_rank": core_rank,
            "peer_outcome": peer_outcome, "peer_status": peer.get("status") if peer else None,
            "peer_rank": peer.get("rank") if peer else None,
            "peer_reason": peer.get("reason") if peer else None,
            "peer_complete": bool(peer and peer.get("complete")),
            "peer_support3_pairs": peer.get("known_pairs_support3") if peer else None,
            "online_peer_status": result.get("stage_forward_aux", {}).get("status") if isinstance(result, dict) else None,
            "profile_start": profile_start, "profile_end": profile_end,
            "profile_start_latency_steps": profile_start - start if profile_start is not None else None,
            "profile_end_latency_steps": profile_end - start if profile_end is not None else None,
            "profile_complete_8_steps": profile_complete,
            "profile_entirely_within_fault": profile_inside,
            "evidence_integrity": evidence,
            "selected_compute_calls": calls, "physical_fma_launches": launches,
            "selected_miss_steps": misses, "scope_trace_drift_steps": drift,
            "scope_execution_valid_steps": valid_steps, "truth_rows": len(truth),
            "truth_path": str(truth_path.relative_to(ROOT)) if truth_path else None,
            "injection_valid": injection_valid,
            "preceding_healthy_interval": baseline_interval,
            "healthy_nonprofile_steps": len(healthy_wall), "fault_nonprofile_steps": len(fault_wall),
            "healthy_nonprofile_median_ms": healthy_median,
            "fault_nonprofile_median_ms": fault_median,
            "fault_nonprofile_degradation_pct": degradation,
            "end_to_end_success": strict,
        })

    healthy_events = [event for event in detections
                      if int(event["iteration"]) > 12 and int(event["iteration"]) not in fault_steps]
    healthy_rows = []
    for event in healthy_events:
        epoch = int(event["epoch"]) if type(event.get("epoch")) is int else None
        loc = localizations.get(epoch) if epoch is not None else None
        result, predicted, rank = result_for(loc)
        healthy_rows.append({"iteration": event.get("iteration"), "epoch": epoch,
            "source": result.get("source", "dc89_core") if isinstance(result, dict) else None,
            "verdict": upper.verdict_bucket(result), "rank": rank,
            "reason": result.get("reason") if isinstance(result, dict) else None,
            "peer_offline": auxiliary.get(epoch) if epoch is not None else None})
    transition = upper.transition_audit(events, schedules, localizations, all_evidence)
    nonfault_health_steps = [int(row["iteration"]) for row in train
                             if int(row.get("iteration", 0)) > 12
                             and int(row["iteration"]) not in fault_steps]
    return {"layout": layout, "session": session.name,
            "prediction_sha256": data["prediction_sha256"],
            "plan_sha256": sha(HERE / f"formal_{layout}_plan.json"),
            "plan_checks": data["plan_checks"], "cases": case_rows,
            "healthy_false_triggers": len(healthy_events),
            "healthy_false_trigger_verdicts": dict(Counter(row["verdict"] for row in healthy_rows)),
            "healthy_wrong_gpu_localizations": sum(row["verdict"] == "localized" for row in healthy_rows),
            "healthy_trigger_verdicts": healthy_rows,
            "healthy_nonfault_nonwarmup_steps": len(nonfault_health_steps),
            "healthy_nonfault_nonwarmup_nonprofile_steps": sum(
                int(train_by_iter[i].get("profile_epoch", 0) or 0) == 0 for i in nonfault_health_steps),
            "peer_offline_windows": [auxiliary[epoch] for epoch in sorted(auxiliary)],
            "profile_transition_audit": transition}


def summarize(layout_results):
    cases = [row for layout in LAYOUTS for row in layout_results[layout]["cases"]]
    if len(cases) != FIXED_DENOMINATOR:
        raise ValueError("blind8 scoring denominator must remain eight")
    outcomes = Counter(row["outcome"] for row in cases)
    by_family, by_layout = {}, {}
    for family in FAMILIES:
        rows = [row for row in cases if row["family"] == family]
        by_family[family] = {"fixed_denominator": 2,
            "strict_success": sum(row["end_to_end_success"] for row in rows),
            "detected": sum(row["detected_in_fault"] for row in rows),
            "correct_rank": sum(row["outcome"] == "correct_rank" for row in rows),
            "wrong_rank": sum(row["outcome"] == "wrong_rank" for row in rows),
            "rejected": sum(row["outcome"] == "rejected" for row in rows),
            "not_triggered": sum(row["outcome"] == "not_triggered" for row in rows),
            "valid_injection": sum(row["injection_valid"] for row in rows),
            "public_sources": dict(Counter(row["public_localization_source"] for row in rows)),
            "peer_outcomes": dict(Counter(row["peer_outcome"] for row in rows))}
    for layout, result in layout_results.items():
        rows = result["cases"]
        by_layout[layout] = {"fixed_denominator": CASES_PER_LAYOUT,
            "strict_success": sum(row["end_to_end_success"] for row in rows),
            "detected": sum(row["detected_in_fault"] for row in rows),
            "correct_rank": sum(row["outcome"] == "correct_rank" for row in rows),
            "wrong_rank": sum(row["outcome"] == "wrong_rank" for row in rows),
            "rejected": sum(row["outcome"] == "rejected" for row in rows),
            "not_triggered": sum(row["outcome"] == "not_triggered" for row in rows),
            "healthy_false_triggers": result["healthy_false_triggers"],
            "healthy_false_trigger_verdicts": result["healthy_false_trigger_verdicts"],
            "healthy_wrong_gpu_localizations": result["healthy_wrong_gpu_localizations"],
            "healthy_nonfault_nonwarmup_steps": result["healthy_nonfault_nonwarmup_steps"],
            "healthy_nonprofile_steps": result["healthy_nonfault_nonwarmup_nonprofile_steps"],
            "public_sources": dict(Counter(row["public_localization_source"] for row in rows)),
            "correct_public_sources": dict(Counter(row["public_localization_source"] for row in rows
                                                    if row["outcome"] == "correct_rank")),
            "core_outcomes": dict(Counter(row["core_outcome"] for row in rows)),
            "peer_outcomes": dict(Counter(row["peer_outcome"] for row in rows)),
            "peer_complete_windows": sum(row["peer_complete"] for row in rows),
            "valid_injections": sum(row["injection_valid"] for row in rows),
            "profile_transition_events": result["profile_transition_audit"]["count"],
            "profile_transition_passed": result["profile_transition_audit"]["passed"]}
    strict = sum(row["end_to_end_success"] for row in cases)
    latency = {}
    for key in ("detection_latency_steps", "profile_start_latency_steps", "profile_end_latency_steps"):
        values = [int(row[key]) for row in cases if row.get(key) is not None]
        latency[key] = {"n": len(values), "missing": len(cases) - len(values),
                        "min": min(values) if values else None,
                        "median": statistics.median(values) if values else None,
                        "max": max(values) if values else None, "values": values}
    return {"protocol": "peer_priority_component_blind8_score_v1",
        "status": "complete", "fixed_denominator": FIXED_DENOMINATOR,
        "strict_success": strict, "gate_threshold": FIXED_DENOMINATOR,
        "gate_passed": strict == FIXED_DENOMINATOR,
        "gate_rule": "all eight fixed cases require fault-window trigger, correct public rank, complete eight-step in-fault evidence and valid scope/FMA; weak, rejected, missed and invalid cases stay in denominator",
        "detected": sum(row["detected_in_fault"] for row in cases),
        "outcomes": dict(outcomes),
        "public_sources": dict(Counter(row["public_localization_source"] for row in cases)),
        "correct_public_sources": dict(Counter(row["public_localization_source"] for row in cases
                                                if row["outcome"] == "correct_rank")),
        "core_outcomes": dict(Counter(row["core_outcome"] for row in cases)),
        "peer_outcomes": dict(Counter(row["peer_outcome"] for row in cases)),
        "peer_offline_complete_fault_windows": sum(row["peer_complete"] for row in cases),
        "peer_offline_correct": sum(row["peer_outcome"] == "correct_rank" for row in cases),
        "peer_offline_wrong": sum(row["peer_outcome"] == "wrong_rank" for row in cases),
        "healthy_false_triggers": sum(row["healthy_false_triggers"] for row in layout_results.values()),
        "healthy_wrong_gpu_localizations": sum(row["healthy_wrong_gpu_localizations"] for row in layout_results.values()),
        "valid_injections": sum(row["injection_valid"] for row in cases),
        "profile_complete_inside_fault": sum(row["profile_complete_8_steps"]
                                               and row["profile_entirely_within_fault"] for row in cases),
        "latency_steps": latency,
        "profile_transition_events": sum(row["profile_transition_audit"]["count"]
                                          for row in layout_results.values()),
        "profile_transition_passed": sum(row["profile_transition_audit"]["passed"]
                                          for row in layout_results.values()),
        "by_layout": by_layout, "by_family": by_family,
        "prediction_before_truth": True, "cases": cases}


def run():
    bundle = verify_pretruth_bundle()
    prediction_freeze = persist_prediction_first(bundle)
    layout_results = {layout: score_layout(layout, bundle["sessions"][layout], bundle)
                      for layout in LAYOUTS}
    summary = summarize(layout_results)
    summary.update({"prediction_freeze": str((HERE / "scorer_prediction_freeze.json").relative_to(ROOT)),
                    "prediction_freeze_sha256": sha(HERE / "scorer_prediction_freeze.json"),
                    "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
                    "core_model_sha256": bundle["freeze"].get("model_sha256"),
                    "peer_model_sha256": bundle["peer_model"].get("sha256"),
                    "prediction_inputs": prediction_freeze,
                    "layout_results": layout_results})
    atomic_json(HERE / "cases.json", {"protocol": summary["protocol"],
                "fixed_denominator": FIXED_DENOMINATOR, "prediction_before_truth": True,
                "layouts": layout_results})
    atomic_json(HERE / "summary.json", summary)
    return summary


def main():
    result = run()
    print(json.dumps({"summary": str((HERE / "summary.json").relative_to(ROOT)),
        "fixed_denominator": result["fixed_denominator"],
        "strict_success": result["strict_success"], "gate_passed": result["gate_passed"],
        "outcomes": result["outcomes"], "healthy_false_triggers": result["healthy_false_triggers"],
        "valid_injections": result["valid_injections"],
        "profile_transition_events": result["profile_transition_events"]}, indent=2))


if __name__ == "__main__":
    main()
