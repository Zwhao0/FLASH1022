#!/usr/bin/env python3
"""CPU-only fixed upper6 L1/L2 plans; no training or prediction work."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import random
import secrets
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
BASE_DIR = ROOT / "results/layer_campaign/continuation_v2/gates"
SOURCE_FREEZE_PATH = HERE.parent / "source_freeze.json"
OUT = {"l1": HERE / "l1", "l2": HERE / "l2"}
LAYOUTS = ("legacy", "node_local_pp")
RANKS = tuple(range(16))
MODEL_PATH = "results/v2_semantic/linear_candidate_model.json"
AUX_MODEL_PATH = "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped.json"
AUX_BINDING_PATH = "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped_context_binding.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from results.layer_campaign.continuation_v2.gates import prepare as base


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def verify_plain2_gate():
    summary_path = HERE.parent / "plain2/score_peer_plain2_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("plain2 integration has no completed independent score")
    summary = read(summary_path)
    if (summary.get("fixed_case_denominator") != 2 or summary.get("strict_cases_passed") != 2
            or summary.get("healthy_holdout_passed") != 2):
        raise RuntimeError("plain2 must pass both fixed cases and both healthy holdouts before upper6 design")
    return {"path": str(summary_path.relative_to(ROOT)), "sha256": sha(summary_path),
            "strict_cases_passed": 2, "healthy_holdout_passed": 2}


def verify_freeze():
    freeze = read(SOURCE_FREEZE_PATH)
    if freeze.get("model_sha256") != base.read_json(ROOT / MODEL_PATH).get("sha256"):
        raise RuntimeError("peer_v1 freeze does not bind the accepted dc89 core")
    for relative, digest in freeze.get("files", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"peer_v1 runtime freeze mismatch: {relative}")
    for relative, digest in freeze.get("binary_files", {}).items():
        if relative != "flash_v1/build/libflashcompute.so":
            raise RuntimeError(f"unexpected frozen binary: {relative}")
        vendor = ROOT / "results/v2_semantic/freeze_inputs/native/libflashcompute.so"
        if sha(vendor) != digest:
            raise RuntimeError("native FMA binary differs from source freeze")
    from flash_v1.stage_forward_peer import verify_model
    model, binding = read(ROOT / AUX_MODEL_PATH), read(ROOT / AUX_BINDING_PATH)
    verify_model(model)
    if binding.get("model_sha256") != model.get("sha256"):
        raise RuntimeError("scoped peer model binding mismatch")
    if model.get("instrumentation_context") != "scoped_dispatch":
        raise RuntimeError("upper6 requires the scoped-dispatch peer model")
    return freeze, model, binding


def make_cases(stage, layout, rng, old_design, attestation):
    baseline = base.old_whole_baseline(old_design, layout)
    topology_ref, topology = base.topology_by_rank(old_design, layout)
    if stage == "l1":
        targets = [(rng.choice(RANKS), "whole")]
    else:
        # One randomized FWD and one randomized BWD target, on distinct ranks.
        fwd = rng.choice(RANKS)
        bwd = rng.choice([rank for rank in RANKS if rank != fwd])
        targets = [(fwd, "fwd"), (bwd, "bwd")]
    cases = []
    for rank, phase in targets:
        target_fraction = rng.uniform(0.40, 0.80)
        dose, catalog = base.dose_for(layout, rank, phase, target_fraction, baseline, attestation, rng)
        case_id = f"upper6-{stage}-{layout}-{phase}-{rank}-{rng.getrandbits(32):08x}"
        fault_target = {"kind": "whole"} if phase == "whole" else {"kind": "stage", "phase": phase}
        row = {"case_id": case_id, "layout": layout, "layer": stage.upper(),
               "category": "whole_rank_compute" if stage == "l1" else "phase_compute",
               "rank": rank, "phase": None if phase == "whole" else phase,
               "population_size": 16, "population_basis": "all validated 16-rank catalog entries; fixed draw before formal",
               "fault_target": fault_target, "rank_catalog": catalog, "dose": dose,
               "training_seed": None, "topology": topology[rank],
               "topology_source": {"path": topology_ref["path"], "sha256": topology_ref["sha256"]}}
        cases.append(row)
    return cases


def intervals_for(stage, cases, mode):
    intervals = [{"id": "warmup", "start": 1, "end": 12, "record_start": 1,
                  "target_ranks": [], "slowdown_extra_ratio": 0.0,
                  "fault_mode": None, "fault_scope": None, "segment": "warmup"},
                 {"id": "healthy_gate", "start": 13, "end": 42, "record_start": 13,
                  "target_ranks": [], "slowdown_extra_ratio": 0.0,
                  "fault_mode": None, "fault_scope": None, "segment": "healthy"}]
    step = 43
    if mode == "timeonly":
        for case in cases:
            intervals.append({"id": f"healthy_before_{case['case_id']}", "start": step, "end": step + 4,
                              "record_start": step, "target_ranks": [], "slowdown_extra_ratio": 0.0,
                              "fault_mode": None, "fault_scope": None, "segment": "healthy"})
            step += 5
            intervals.append({"id": case["case_id"], "case_id": case["case_id"], "start": step, "end": step + 4,
                              "record_start": step, "target_ranks": [case["rank"]], "rank": case["rank"],
                              "phase": case.get("phase"), "slowdown_extra_ratio": case["dose"]["slowdown_extra_ratio"],
                              "fault_mode": "whole_compute", "fault_scope": case["fault_target"],
                              "segment": "fault", "fault_enabled": True})
            step += 5
        intervals.append({"id": "healthy_tail", "start": step, "end": step + 4, "record_start": step,
                          "target_ranks": [], "slowdown_extra_ratio": 0.0,
                          "fault_mode": None, "fault_scope": None, "segment": "healthy"})
    else:
        for case in cases:
            intervals.append({"id": case["case_id"], "case_id": case["case_id"], "start": step, "end": step + 31,
                              "record_start": step, "target_ranks": [case["rank"]], "rank": case["rank"],
                              "phase": case.get("phase"), "slowdown_extra_ratio": case["dose"]["slowdown_extra_ratio"],
                              "fault_mode": "whole_compute", "fault_scope": case["fault_target"],
                              "segment": "fault", "fault_enabled": True})
            step += 32
            intervals.append({"id": f"healthy_after_{case['case_id']}", "start": step, "end": step + 29,
                              "record_start": step, "target_ranks": [], "slowdown_extra_ratio": 0.0,
                              "fault_mode": None, "fault_scope": None, "segment": "healthy"})
            step += 30
        intervals.append({"id": "healthy_tail", "start": step, "end": step + 14, "record_start": step,
                          "target_ranks": [], "slowdown_extra_ratio": 0.0,
                          "fault_mode": None, "fault_scope": None, "segment": "healthy"})
    expected = 42 + len(cases) * 10 + 5 if mode == "timeonly" else 42 + len(cases) * 62 + 15
    if intervals[-1]["end"] != expected:
        raise AssertionError((stage, mode, len(cases), intervals[-1]["end"], expected))
    return intervals, expected


def make_plan(stage, layout, mode, cases, design_sha, source_sha, design, model):
    intervals, steps = intervals_for(stage, cases, mode)
    core_model = read(ROOT / MODEL_PATH)
    for case in cases:
        case["training_seed"] = design["training_seed"]
    plan = {"schema_version": 1, "protocol": f"upper6_{stage}_{mode}_v1",
            "session_id": f"v2s_peer_upper6_{stage}_{mode}_{layout}_01",
            "layout": layout, "parallel_layout": layout, "seed": design["training_seed"],
            "steps": steps, "seq_len": 16, "microbatches": 2, "warmup_steps": 12,
            "online": mode == "formal", "probe_enabled": False, "profile_enabled": False,
            "evidence_enabled": mode == "formal", "measurement_only": mode == "timeonly",
            "max_profiles": 0 if mode == "timeonly" else 16,
            "role": f"upper6_{stage}_{mode}", "execution_role": mode,
            "fault_kind": "whole_rank_compute" if stage == "l1" else "phase_compute",
            "scoped_compute_enabled": True, "whole_compute_reference_session": f"v2s_scope_catalog_{layout}_02",
            "scoped_reference_session": f"v2s_scope_catalog_{layout}_02",
            "design_sha256": design_sha, "freeze_manifest_sha256": source_sha,
            "fixed_denominator": 2 if stage == "l1" else 4,
            "monitor_resume_policy": "off_by_default", "monitor_confirm_recent_three": True,
            "monitor_profile_transition": mode == "formal", "retain_all_cases": True,
            "retain_all_false_alarms": True,
            "stage_forward_aux_enabled": mode == "formal",
            "stage_forward_aux_model": AUX_MODEL_PATH,
            "stage_forward_aux_context_binding": AUX_BINDING_PATH,
            "stage_forward_aux_model_file_sha256": sha(ROOT / AUX_MODEL_PATH),
            "stage_forward_aux_binding_file_sha256": sha(ROOT / AUX_BINDING_PATH),
            "controlled_profile_requests": [], "intervals": intervals}
    if mode == "formal":
        plan.update({"profile_backend": "semantic_only", "evidence_model": MODEL_PATH,
                     "evidence_model_file_sha256": sha(ROOT / MODEL_PATH),
                     "frozen_model_sha256": core_model["sha256"], "detector_and_localizer_frozen": True})
    return plan


def make_timeonly(stage):
    out = OUT[stage]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"preserve existing upper6 artifacts: {out}")
    plain2 = verify_plain2_gate()
    source_freeze, model, _ = verify_freeze()
    out.mkdir(parents=True, exist_ok=True)
    old_path = ROOT / base.OLD_DESIGN_PATH
    old_design = read(old_path)
    att, _ = base.attestation_inputs(old_design)
    # Validate all 16 rank catalogs before drawing sites or dose.
    for layout in LAYOUTS:
        for rank in RANKS:
            base.catalog_for(att, layout, rank)
    design_seed_hex = secrets.token_hex(16)
    rng = random.Random(int(design_seed_hex, 16))
    training_seed = rng.randrange(1, 2**31 - 1)
    design = {"protocol": f"upper6_{stage}_design_v1", "stage": stage.upper(),
              "integration_not_blind": True, "design_seed_hex": design_seed_hex,
              "training_seed": training_seed, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_freeze": {"path": str(SOURCE_FREEZE_PATH.relative_to(ROOT)), "sha256": sha(SOURCE_FREEZE_PATH),
                                "dc89_sha256": source_freeze["model_sha256"]},
              "plain2_gate": plain2, "peer_model_sha256": read(ROOT / AUX_MODEL_PATH)["sha256"],
              "fixed_denominator": 2 if stage == "l1" else 4,
              "cases_per_layout": 1 if stage == "l1" else 2,
              "dose_policy": "independent rank draws and per-case U(0.40,0.80) target; scope catalog estimate; one time-only pass; no outcome-based replacement or retuning",
              "schedule": {"timeonly_steps_per_layout": 57 if stage == "l1" else 67,
                           "formal_steps_per_layout": 119 if stage == "l1" else 181,
                           "timeonly": "12 warmup + 30 stable-health gate + 5 healthy + 5 fault per case + 5 tail",
                           "formal": "42 healthy opening + 32 fault + 30 healthy per case + 15 tail"},
              "layouts": {}}
    design_sha_path = out / "design.json"
    for layout in LAYOUTS:
        cases = make_cases(stage, layout, rng, old_design, att)
        for case in cases:
            case["training_seed"] = training_seed
        design["layouts"][layout] = cases
    atomic_json(design_sha_path, design)
    design_sha = sha(design_sha_path)
    source_sha = sha(SOURCE_FREEZE_PATH)
    for layout in LAYOUTS:
        plan = make_plan(stage, layout, "timeonly", design["layouts"][layout], design_sha, source_sha, design, model)
        atomic_json(out / f"timeonly_{layout}_plan.json", plan)
    artifact_names = ["design.json", "timeonly_legacy_plan.json", "timeonly_node_local_pp_plan.json"]
    ready = {"status": "timeonly_ready_for_single_gpu_owner", "stage": stage.upper(),
             "created_utc": datetime.now(timezone.utc).isoformat(), "fixed_denominator": design["fixed_denominator"],
             "source_freeze_sha256": source_sha, "design_sha256": design_sha,
             "artifacts": {name: sha(out / name) for name in artifact_names}}
    ready["artifacts"]["source_freeze.json"] = source_sha
    atomic_json(out / "timeonly_ready.json", ready)
    return ready


def finalize(stage):
    out = OUT[stage]
    gate_path = out / "timeonly_gate.json"
    if not gate_path.is_file() or read(gate_path).get("stable_for_formal") is not True:
        raise RuntimeError("stable 30-step time-only gate required before formal plans")
    ready0 = read(out / "timeonly_ready.json")
    if sha(SOURCE_FREEZE_PATH) != ready0["source_freeze_sha256"]:
        raise RuntimeError("peer source freeze changed after draw")
    for name, digest in ready0["artifacts"].items():
        path = SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name
        if sha(path) != digest:
            raise RuntimeError(f"time-only input changed: {name}")
    design = read(out / "design.json")
    _, model, _ = verify_freeze()
    for layout in LAYOUTS:
        plan = make_plan(stage, layout, "formal", design["layouts"][layout], sha(out / "design.json"),
                         sha(SOURCE_FREEZE_PATH), design, model)
        atomic_json(out / f"formal_{layout}_plan.json", plan)
    names = ["source_freeze.json", "timeonly_gate.json", "design.json", "timeonly_ready.json",
             "formal_legacy_plan.json", "formal_node_local_pp_plan.json"]
    artifacts = {name: sha(SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name) for name in names}
    ready = {"status": "ready_for_single_gpu_owner", "stage": stage.upper(),
             "created_utc": datetime.now(timezone.utc).isoformat(), "fixed_denominator": design["fixed_denominator"],
             "source_freeze_sha256": sha(SOURCE_FREEZE_PATH), "design_sha256": sha(out / "design.json"),
             "timeonly_gate_sha256": sha(gate_path), "artifacts": artifacts}
    atomic_json(out / "plan_ready.json", ready)
    return ready


def repair_unstarted_timeonly(stage):
    """Preserve and minimally correct a published bundle before any runner starts."""
    out = OUT[stage]
    names = ["design.json", "timeonly_legacy_plan.json", "timeonly_node_local_pp_plan.json", "timeonly_ready.json"]
    if not all((out / name).is_file() for name in names):
        raise FileNotFoundError("complete prelaunch timeonly bundle required")
    pre = out / "prelaunch_invalid_metadata"
    if pre.exists():
        raise FileExistsError(f"preserve existing prelaunch amendment: {pre}")
    pre.mkdir(parents=True)
    old_ready = read(out / "timeonly_ready.json")
    old_hashes = {name: sha(out / name) for name in names}
    for name in names:
        shutil.copy2(out / name, pre / name)

    freeze = read(SOURCE_FREEZE_PATH)
    design = read(out / "design.json")
    design["source_freeze"]["dc89_sha256"] = freeze["model_sha256"]
    design["peer_model_sha256"] = read(ROOT / AUX_MODEL_PATH)["sha256"]
    atomic_json(out / "design.json", design)
    design_sha = sha(out / "design.json")
    for layout in LAYOUTS:
        plan = read(out / f"timeonly_{layout}_plan.json")
        plan["design_sha256"] = design_sha
        plan["stage_forward_aux_enabled"] = False
        atomic_json(out / f"timeonly_{layout}_plan.json", plan)

    amendment = {"protocol": "upper6_prelaunch_metadata_correction_v1",
                 "status": "corrected_before_execution", "stage": stage.upper(),
                 "reason": ["keep dc89 core hash separate from auxiliary TP-peer hash",
                            "do not enable stage auxiliary during measurement_only timeonly plans"],
                 "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
                 "training_seed_unchanged": design["training_seed"],
                 "case_ids_unchanged": {layout: [case["case_id"] for case in design["layouts"][layout]]
                                        for layout in LAYOUTS},
                 "rank_dose_unchanged": True, "runner_started": False,
                 "prior_ready_sha256": old_hashes["timeonly_ready.json"],
                 "prior_artifacts": old_hashes}
    atomic_json(out / "prelaunch_amendment.json", amendment)
    artifacts = {"design.json": sha(out / "design.json"),
                 "timeonly_legacy_plan.json": sha(out / "timeonly_legacy_plan.json"),
                 "timeonly_node_local_pp_plan.json": sha(out / "timeonly_node_local_pp_plan.json"),
                 "prelaunch_amendment.json": sha(out / "prelaunch_amendment.json"),
                 "source_freeze.json": sha(SOURCE_FREEZE_PATH)}
    ready = dict(old_ready)
    ready.update({"status": "timeonly_ready_for_single_gpu_owner",
                  "design_sha256": design_sha, "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
                  "prelaunch_amendment_sha256": artifacts["prelaunch_amendment.json"],
                  "artifacts": artifacts})
    atomic_json(out / "timeonly_ready.json", ready)
    return ready


def check():
    for stage, per_layout, timeonly, formal in (("l1", 1, 57, 119), ("l2", 2, 67, 181)):
        dummy = [{"case_id": f"case-{i}", "rank": i, "phase": ("fwd" if i == 0 else "bwd"),
                  "fault_target": {"kind": "whole"} if stage == "l1" else {"kind": "stage", "phase": "fwd" if i == 0 else "bwd"},
                  "dose": {"slowdown_extra_ratio": 1.0}} for i in range(per_layout)]
        assert intervals_for(stage, dummy, "timeonly")[1] == timeonly
        assert intervals_for(stage, dummy, "formal")[1] == formal
    print("upper6 schedule check passed; this check draws no seed and writes no design")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs="?", choices=("make-timeonly", "finalize", "repair-unstarted-timeonly"))
    parser.add_argument("stage", nargs="?", choices=("l1", "l2"))
    args = parser.parse_args()
    if args.check:
        check()
        return
    if not args.command or args.stage is None:
        parser.error("stage l1 or l2 is required")
    if args.command == "make-timeonly":
        value = make_timeonly(args.stage)
    elif args.command == "repair-unstarted-timeonly":
        value = repair_unstarted_timeonly(args.stage)
    else:
        value = finalize(args.stage)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
