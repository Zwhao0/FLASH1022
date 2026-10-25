#!/usr/bin/env python3
"""CPU-only independent score for the fixed production no-diagnostic quick4."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_reverted_baseline"
CANDIDATE = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_sparse3_gc_lifetime_v2/score_candidate.py"
LAYOUTS = ("group1", "group2", "group3")


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if "--sealed-owner-confirmed" not in sys.argv:raise SystemExit("requires all three group public snapshots sealed")
    load(HERE / "run.py", "quick_runner_check").check(live=True)
    assert read(HERE / "scoring_rule_before_truth.json")["scorer_sha256"]==sha(Path(__file__))
    contract = read(HERE / "contract.json")
    freeze = read(HERE / "source_freeze.json")
    complete = read(HERE / "execution_complete.json")
    index = read(HERE / "public_predictions_before_truth.json")
    freeze_sha = sha(HERE / "source_freeze.json")
    if (complete.get("status") != "complete" or complete.get("fixed_cases") != 6
            or complete.get("source_freeze_sha256") != freeze_sha
            or index.get("prediction_before_truth") is not True
            or index.get("fixed_cases") != 6):
        raise ValueError("quick4 completion/public-freeze contract mismatch")
    # Verify both immutable public snapshots and source-bound plans before reading truth.
    assert set(index["sessions"])==set(LAYOUTS) and set(complete["sessions"])==set(LAYOUTS)
    data, design = {}, {"layouts": {}}
    cases_by_group = {layout: [] for layout in LAYOUTS}
    for case in contract["cases"]:
        f=case["fault_interval"]
        cases_by_group[case["group"]].append(dict(case_id=f["case_id"],parent_case_id=f["case_id"],regression_family="whole_compute" if case["tier"]=="L1" else "backward_stage" if case["phase"]=="bwd" else "forward_stage",rank=case["rank"],phase=case.get("phase"),fault_target=case["fault_target"],operator_family=None))
    for layout in LAYOUTS:
        plan_info = contract["plans"][layout]
        plan_path = ROOT / plan_info["path"]
        plan = read(plan_path)
        if sha(plan_path) != plan_info["sha256"] or plan.get("steps") != plan_info["steps"]:
            raise ValueError(f"{layout}: frozen plan hash/length mismatch")
        session = ROOT / complete["sessions"][layout]["path"]
        if not session.is_dir():
            raise FileNotFoundError(session)
        for host, node in (("host1", 1), ("host2", 0)):
            manifest = read(session / host / f"node{node}_manifest.json")
            if manifest.get("status") != "completed":
                raise ValueError(f"{layout}/{host}: node not completed")
            actual_plan = read(session / host / "plan.json")
            if actual_plan.get("session_id") != plan["session_id"] or actual_plan.get("intervals") != plan["intervals"]:
                raise ValueError(f"{layout}/{host}: persisted fault schedule differs")
        events_path = session / "host2/events.jsonl"
        public_events = [json.loads(line) for line in (events_path.read_text().splitlines() if events_path.exists() else [])
                         if json.loads(line).get("event") in {"detection", "profile_schedule", "localization", "quota_exhausted"}]
        pred_path = HERE / f"{layout}_predictions_without_truth.json"
        pred = read(pred_path)
        if (pred.get("prediction_before_truth") is not True
                or pred.get("session_id") != plan["session_id"]
                or pred.get("plan_sha256") != sha(plan_path)
                or pred.get("events_sha256") != (sha(events_path) if events_path.exists() else None)
                or pred.get("events") != public_events
                or index.get("sessions", {}).get(layout, {}).get("prediction_sha256") != sha(pred_path)):
            raise ValueError(f"{layout}: public prediction snapshot mismatch")
        data[layout] = {
            "session": session,
            "plan": plan,
            "events": public_events,
            "prediction_sha256": sha(pred_path),
            "plan_sha256": sha(plan_path),
        }
        design["layouts"][layout] = cases_by_group[layout]

    scorer = load(OLD / "score.py", "quick4_baseline_score")
    candidate = load(CANDIDATE, "quick4_sparse_candidate_score")
    candidate.install_sparse_auxiliary_check(scorer)
    # Only the new frozen protocol duration differs: 24 actual injected steps.
    # Core eight-step/512+128 and sparse48 validity gates remain unchanged.
    import inspect
    source=inspect.getsource(scorer.score_layout)
    for old,new in [("aux.get(\"rows\") == 128 and aux.get(\"expected_rows\") == 128","aux.get(\"rows\") == 48 and aux.get(\"expected_rows\") == 48"),("len(truth) == 32","len(truth) == 24"),("valid_steps == 32","valid_steps == 24"),("case[\"case_id\"]: 32","case[\"case_id\"]: 24")]:
        assert old in source,old
        source=source.replace(old,new)
    namespace=dict(scorer.__dict__);exec(compile(source,"quick24-injection-duration-only","exec"),namespace);scorer.score_layout=namespace["score_layout"]
    bundle = {"design": design}
    # Only after both public snapshots above have been verified do these calls read truth.
    layouts = {group: scorer.score_layout(contract["plans"][group]["layout"], data[group], {"design":{"layouts":{contract["plans"][group]["layout"]:cases_by_group[group]}}}) for group in LAYOUTS}
    enrich(layouts,data,scorer)
    cases = [row for layout in LAYOUTS for row in layouts[layout]["cases"]]
    if len(cases) != 6:
        raise ValueError("quick4 fixed case denominator changed")
    summary_cases = []
    for row in cases:
        summary_cases.append({k: row.get(k) for k in (
            "case_id", "group", "layout", "tier", "family", "frozen_dose", "rank", "fault_start", "fault_end",
            "detection_iteration", "detection_latency_steps", "public_outcome", "public_rank",
            "core_outcome", "core_rank", "peer_outcome", "peer_rank", "peer_reason",
            "profile_start", "profile_end", "profile_complete_8_steps", "profile_entirely_within_fault",
            "semantic_evidence_integrity", "auxiliary_integrity_passed",
            "auxiliary_comparable_tp_pairs_support3", "truth_rows", "selected_compute_calls",
            "physical_fma_launches", "selected_miss_steps", "scope_trace_drift_steps",
            "scope_execution_valid_steps", "injection_valid", "end_to_end_success", "truth_path", "detected_in_fault", "first_actual_accepted_epoch", "first_actual_accepted_rank", "first_actual_accepted_outcome", "detection_lineage", "observed_nonprofile_wall_degradation_pct", "healthy_reference_nonprofile_steps", "fault_nonprofile_steps")})
    summary = {
        "protocol": "production_no_diag_quick_l1l2_three_groups_six_24step_v1",
        "not_a_new_campaign_accuracy_estimate": True,
        "fixed_denominator": 6,
        "source_freeze_sha256": freeze_sha,
        "execution_complete_sha256": sha(HERE / "execution_complete.json"),
        "public_prediction_index_sha256": sha(HERE / "public_predictions_before_truth.json"),
        "source_freeze_status": freeze.get("status"),
        "strict_success": sum(row.get("end_to_end_success") is True for row in cases),
        "public_correct": sum(row.get("public_outcome") == "correct_rank" for row in cases),
        "core_correct": sum(row.get("core_outcome") == "correct_rank" for row in cases),
        "peer_correct": sum(row.get("peer_outcome") == "correct_rank" for row in cases),
        "healthy_false_triggers": sum(v["healthy_false_triggers"] for v in layouts.values()),
        "healthy_wrong_gpu_localizations": sum(v["healthy_wrong_gpu_localizations"] for v in layouts.values()),
        "healthy_steps": sum(v["healthy_nonfault_nonwarmup_steps"] for v in layouts.values()),
        "by_group": {layout: {
            "fixed_denominator": len(result["cases"]),
            "strict_success": sum(row.get("end_to_end_success") is True for row in result["cases"]),
            "healthy_false_triggers": result["healthy_false_triggers"],
            "healthy_wrong_gpu_localizations": result["healthy_wrong_gpu_localizations"],
            "healthy_nonfault_nonwarmup_steps": result["healthy_nonfault_nonwarmup_steps"],
        } for layout, result in layouts.items()},
        "cases": summary_cases,
    }
    summary["first_actual_accepted_correct"] = sum(r["first_actual_accepted_outcome"]=="correct_rank" for r in cases)
    summary["first_actual_accepted_wrong"] = sum(r["first_actual_accepted_outcome"]=="wrong_rank" for r in cases)
    summary["fault_duration_steps"] = 24
    summary["layouts_full_audit"] = layouts
    summary["healthy_wrong_gpu_raw_detection_time_count"] = summary["healthy_wrong_gpu_localizations"]
    summary["pure_healthy_profile_wrong_gpu_localizations"] = sum(r["pure_healthy_profile_localized_count"] for r in layouts.values())
    summary["gate_passed"] = summary["strict_success"] == 6 and summary["pure_healthy_profile_wrong_gpu_localizations"] == 0
    out = HERE / "score_summary.json"
    if out.exists():
        if read(out) != summary:
            raise FileExistsError(f"preserve different existing score artifact: {out}")
    else:
        out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    brief={k:v for k,v in summary.items() if k!="layouts_full_audit"}
    brief["health_by_group"]={name:{k:r[k] for k in ("healthy_detection_candidates","healthy_nonfault_nonwarmup_steps","healthy_monitor_steps_without_profile","healthy_detection_profile_overlaps_fault_count","pure_healthy_profile_count","pure_healthy_profile_localized_count","healthy_trigger_verdicts")} for name,r in layouts.items()}
    brief_path=HERE/"score_brief.json"
    if brief_path.exists():assert read(brief_path)==brief
    else:brief_path.write_text(json.dumps(brief,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k: summary[k] for k in (
        "strict_success", "fixed_denominator", "public_correct", "core_correct",
        "peer_correct", "healthy_false_triggers", "healthy_wrong_gpu_localizations", "healthy_steps", "gate_passed")}
        | {"score_summary": str(out.relative_to(ROOT))}, indent=2))


def enrich(layouts, data, scorer):
    import statistics
    frozen_cases = {r["fault_interval"]["case_id"]: r for r in read(HERE / "contract.json")["cases"]}
    for layout, result in layouts.items():
        plan, session, events = (data[layout][k] for k in ("plan", "session", "events"))
        train = scorer.read_jsonl(session / "host2/train.jsonl")
        by_iter = {int(r["iteration"]): r for r in train}
        faults = [r for r in plan["intervals"] if r.get("target_ranks")]
        fault_steps = {i for f in faults for i in range(f["start"], f["end"] + 1)}
        schedules = {r["epoch"]: r for r in events if r.get("event") == "profile_schedule"}
        healthy = result["healthy_trigger_verdicts"]
        for row in healthy:
            profile = schedules.get(row["epoch"])
            row["profile_start"] = profile["start"] if profile else None
            row["profile_end"] = profile["end"] if profile else None
            row["profile_overlaps_fault"] = bool(profile and any(i in fault_steps for i in range(profile["start"], profile["end"] + 1)))
            row["overlapping_faults"] = [{"case_id": f["case_id"], "rank": f["target_ranks"][0], "start": f["start"], "end": f["end"]} for f in faults if profile and max(f["start"], profile["start"]) <= min(f["end"], profile["end"])]
            row["public_rank_matches_overlapping_fault"] = any(row["rank"] == f["rank"] for f in row["overlapping_faults"])
            row["profile_fully_in_fault"] = bool(profile and any(f["start"] <= profile["start"] and profile["end"] <= f["end"] for f in faults))
            row["pure_healthy_profile"] = bool(profile and profile["start"] > 12 and not row["profile_overlaps_fault"])
        result["healthy_detection_candidates"] = len(healthy)
        result["healthy_detection_profile_overlaps_fault_count"] = sum(r["profile_overlaps_fault"] for r in healthy)
        result["pure_healthy_profile_count"] = sum(r["pure_healthy_profile"] for r in healthy)
        result["pure_healthy_profile_localized_count"] = sum(r["pure_healthy_profile"] and r["verdict"] == "localized" for r in healthy)
        result["healthy_monitor_steps_without_profile"] = sum(i > 12 and i not in fault_steps and not int(r.get("profile_epoch", 0) or 0) for i, r in by_iter.items())
        result["detection_source_counts"] = dict(scorer.Counter(r.get("source", "unspecified") for r in events if r.get("event") == "detection"))
        for case in result["cases"]:
            detections=sorted((e for e in events if e.get("event")=="detection" and case["fault_start"]<=e.get("iteration",-1)<=case["fault_end"]),key=lambda e:(e["iteration"],e.get("epoch",0)))
            localizations={e["epoch"]:e for e in events if e.get("event")=="localization"}
            accepted=[]
            case["detection_lineage"]=[{k:e.get(k) for k in ("iteration","epoch","source","parent_epoch")} for e in detections]
            for detection in detections:
                loc=localizations.get(detection.get("epoch"));public,_,rank=scorer.result_fields(loc)
                if public.get("status")=="localized":accepted.append((detection.get("epoch"),rank))
            case["first_actual_accepted_epoch"]=accepted[0][0] if accepted else None
            case["first_actual_accepted_rank"]=accepted[0][1] if accepted else None
            case["first_actual_accepted_outcome"]="not_accepted" if not accepted else "correct_rank" if accepted[0][1]==case["rank"] else "wrong_rank"
            frozen = frozen_cases[case["case_id"]]
            case.update(group=frozen["group"],tier=frozen["tier"], drawn_family=frozen.get("family"), frozen_dose=frozen["dose"], frozen_slowdown_extra_ratio=frozen["fault_interval"]["slowdown_extra_ratio"])
            start, end = case["fault_start"], case["fault_end"]
            before = next((p for p in reversed(plan["intervals"]) if p.get("segment") == "healthy" and p["end"] < start), None)
            after = next((p for p in plan["intervals"] if p.get("segment") == "healthy" and p["start"] > end), None)
            def valid_iters(a, b):
                return [i for i in range(max(13, a), b + 1) if i in by_iter and not int(by_iter[i].get("profile_epoch", 0) or 0)]
            hi_before = valid_iters(before["start"], before["end"])[-16:] if before else []
            hi_after = valid_iters(after["start"], after["end"])[:16] if after else []
            fi = valid_iters(start, end)
            ht = [float(by_iter[i]["wall_iteration_ms"]) for i in hi_before + hi_after]
            ft = [float(by_iter[i]["wall_iteration_ms"]) for i in fi]
            hm, fm = (statistics.median(x) if x else None for x in (ht, ft))
            case.update(healthy_reference_before_steps=hi_before, healthy_reference_after_steps=hi_after, healthy_reference_nonprofile_steps=len(ht), fault_nonprofile_steps=len(ft), healthy_nonprofile_median_ms=hm, fault_nonprofile_median_ms=fm, observed_nonprofile_wall_degradation_pct=100 * (fm / hm - 1) if hm and fm else None)
            case["degradation_interpretation"] = "host2 rank0 merged train wall_iteration_ms; median of all valid fault non-PROFILE steps vs pooled nearest up-to-16 valid non-PROFILE steps in each adjacent healthy interval, excluding warmup; fixed rule before truth, no window selection; not PROFILE overhead or causal isolated compute cost"
            case["extra_in_fault_detections"] = max(0, case["detection_count_in_fault"] - 1)
            case["in_fault_detection_source_counts"] = dict(scorer.Counter(r.get("source", "unspecified") for r in events if r.get("event") == "detection" and start <= r.get("iteration", -1) <= end))

if __name__ == "__main__":
    main()
