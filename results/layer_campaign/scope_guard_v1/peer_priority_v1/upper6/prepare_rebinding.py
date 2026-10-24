#!/usr/bin/env python3
"""Prepare the inherited L1/L2 upper6 regression on the peer-priority freeze.

CPU-only: copies the completed guarded plans and changes only their session and
freeze bindings. It never reads blind8 runtime output or starts training.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "results/layer_campaign/scope_guard_v1"
PARENT = BASE / "revalidation"
PEER = BASE / "peer_priority_v1"
OUT = PEER / "upper6"
CURRENT_FREEZE = PEER / "source_freeze.json"
ORIGINAL_FREEZE = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/source_freeze.json"
REMOTE_CHECK = PEER / "remote_source_check.json"
LAYOUTS = ("legacy", "node_local_pp")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def scorer_with_rebinding(base_score: str) -> str:
    constants = {
        'SOURCE_FREEZE_PATH = HERE.parent / "source_freeze.json"':
            'SOURCE_FREEZE_PATH = HERE.parent / "source_freeze.json"',
        'PARENT_SOURCE_FREEZE_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/source_freeze.json"':
            'PARENT_SOURCE_FREEZE_PATH = HERE / "parent_guard_source_freeze.json"\nPARENT_CORE_SOURCE_FREEZE_PATH = HERE / "parent_core_source_freeze.json"',
        'ORIGINAL_STAGE_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/upper6"':
            'ORIGINAL_STAGE_PATH = ROOT / "results/layer_campaign/scope_guard_v1/revalidation"\nORIGINAL_UPPER6_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/upper6"',
    }
    for old, new in constants.items():
        if old not in base_score:
            raise RuntimeError(f"score template constant missing: {old}")
        base_score = base_score.replace(old, new, 1)

    start = base_score.index("def verify_bound_inputs(")
    end = base_score.index("\ndef compact_timeline", start)
    verifier = r'''def verify_bound_inputs(stage, layout, prediction, out):
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
'''
    base_score = base_score[:start] + verifier + base_score[end:]
    return base_score


def main():
    if any((OUT / stage).exists() for stage in ("l1", "l2")):
        raise RuntimeError(f"refusing to overwrite existing upper6 stage bundle: {OUT}")
    current_freeze, guard_freeze, core_freeze = read(CURRENT_FREEZE), read(PARENT / "source_freeze.json"), read(ORIGINAL_FREEZE)
    remote = read(REMOTE_CHECK)
    if current_freeze.get("parent_source_freeze", {}).get("sha256") != sha(PARENT / "source_freeze.json"):
        raise RuntimeError("peer-priority source does not descend from the revalidation freeze")
    if len({current_freeze.get("model_sha256"), guard_freeze.get("model_sha256"),
            core_freeze.get("model_sha256")}) != 1:
        raise RuntimeError("core model hash changed across this source-only rebind")
    if not remote or any(e.get("passed") is not True or e.get("files") != current_freeze.get("files")
                         for e in remote.values()):
        raise RuntimeError("peer-priority remote source check is incomplete or mismatched")
    changed_files = sorted(k for k in set(guard_freeze["files"]) | set(current_freeze["files"])
                           if guard_freeze["files"].get(k) != current_freeze["files"].get(k))
    if changed_files != ["flash_v1/online.py", "flash_v1/online_v2.py"]:
        raise RuntimeError(f"unexpected source delta: {changed_files}")

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "source_freeze.json").exists():
        if sha(OUT / "source_freeze.json") != sha(CURRENT_FREEZE):
            raise RuntimeError("upper6 root already binds a different source freeze")
    else:
        shutil.copy2(CURRENT_FREEZE, OUT / "source_freeze.json")
    scorer_template = (PARENT / "l1/score.py").read_text()
    scorer_source = scorer_with_rebinding(scorer_template)
    for stage, expected in (("l1", 2), ("l2", 4)):
        source_dir, dest = PARENT / stage, OUT / stage
        dest.mkdir()
        # Copy the immutable parent design/gate lineage, not runtime output.
        for filename in ("design.json", "timeonly_gate.json", "timeonly_ready.json", "revalidation_amendment.json"):
            shutil.copy2(source_dir / filename, dest / filename)
        shutil.copy2(source_dir / "plan_ready.json", dest / "parent_plan_ready.json")
        shutil.copy2(PARENT / "source_freeze.json", dest / "parent_guard_source_freeze.json")
        shutil.copy2(ORIGINAL_FREEZE, dest / "parent_core_source_freeze.json")
        shutil.copy2(REMOTE_CHECK, dest / "remote_source_check.json")

        parent_plans, current_plans, layouts_record = {}, {}, {}
        for layout in LAYOUTS:
            parent_path = source_dir / f"formal_{layout}_plan.json"
            parent_plan = read(parent_path)
            parent_copy = dest / f"parent_formal_{layout}_plan.json"
            shutil.copy2(parent_path, parent_copy)
            plan = dict(parent_plan)
            plan["session_id"] = f"v2s_peer_priority_upper6_{stage}_{layout}_01"
            plan["freeze_manifest_sha256"] = sha(CURRENT_FREEZE)
            plan_path = dest / f"formal_{layout}_plan.json"
            write(plan_path, plan)
            changed = sorted(k for k in set(parent_plan) | set(plan) if parent_plan.get(k) != plan.get(k))
            if changed != ["freeze_manifest_sha256", "session_id"]:
                raise RuntimeError(f"unexpected plan mutation at {stage}/{layout}: {changed}")
            parent_plans[layout], current_plans[layout] = parent_plan, plan
            layouts_record[layout] = {
                "session_id": plan["session_id"], "parent_plan_sha256": sha(parent_copy),
                "new_plan_sha256": sha(plan_path), "changed_fields": changed,
                "preserved_fields_sha256": canonical_sha({k: v for k, v in plan.items()
                                                            if k not in ("session_id", "freeze_manifest_sha256")}),
                "case_ids": [x["case_id"] for x in read(dest / "design.json")["layouts"][layout]],
            }
        if sum(len(v) for v in read(dest / "design.json")["layouts"].values()) != expected:
            raise RuntimeError(f"unexpected inherited fixed denominator for {stage}")

        amendment = {
            "protocol": "peer_priority_upper6_session_freeze_rebinding_v1",
            "status": "ready_for_owner_review",
            "stage": stage.upper(),
            "controlled_regression_not_fresh_acceptance": True,
            "no_redraw_no_timeonly_repeat_no_result_based_selection": True,
            "fixed_denominator": expected,
            "source_freeze_path": "results/layer_campaign/scope_guard_v1/peer_priority_v1/source_freeze.json",
            "source_freeze_sha256": sha(CURRENT_FREEZE),
            "parent_guard_freeze_sha256": sha(PARENT / "source_freeze.json"),
            "parent_core_freeze_sha256": sha(ORIGINAL_FREEZE),
            "parent_revalidation_amendment_sha256": sha(dest / "revalidation_amendment.json"),
            "parent_plan_ready_sha256": sha(dest / "parent_plan_ready.json"),
            "design_sha256": sha(dest / "design.json"),
            "timeonly_gate_sha256": sha(dest / "timeonly_gate.json"),
            "source_delta_from_guard": changed_files,
            "inherited_gate": "The parent stable time-only gate is reused byte-for-byte; no new time-only calibration.",
            "preserved": "Parent cases, seeds, target ranks, doses, intervals, model and runtime flags; only session_id and freeze_manifest_sha256 differ in each formal plan.",
            "layouts": layouts_record,
        }
        write(dest / "upper6_rebinding_amendment.json", amendment)
        (dest / "score.py").write_text(scorer_source)

        artifacts = {}
        for item in sorted(dest.iterdir()):
            if item.is_file() and item.name != "plan_ready.json" and not item.name.endswith(".tmp"):
                artifacts[item.name] = sha(item)
        artifacts["source_freeze.json"] = sha(OUT / "source_freeze.json")
        ready = {
            "protocol": "peer_priority_upper6_inherited_plan_regression_v1",
            "status": "ready_for_single_gpu_owner_after_blind8_gate",
            "stage": stage.upper(),
            "controlled_regression_not_fresh_acceptance": True,
            "fixed_denominator": expected,
            "blind8_launch_gate": "Require completed fixed8 score strict_success=8 and healthy_localized_false_alarms=0 before execution.",
            "source_freeze_sha256": sha(OUT / "source_freeze.json"),
            "parent_guard_freeze_sha256": sha(PARENT / "source_freeze.json"),
            "parent_design_sha256": sha(dest / "design.json"),
            "amendment_sha256": sha(dest / "upper6_rebinding_amendment.json"),
            "remote_source_check_sha256": sha(dest / "remote_source_check.json"),
            "artifacts": artifacts,
        }
        write(dest / "plan_ready.json", ready)

        # Static, CPU-only ready check; never accesses session results or blind8 output.
        for name, digest in artifacts.items():
            path = OUT / "source_freeze.json" if name == "source_freeze.json" else dest / name
            if sha(path) != digest:
                raise RuntimeError(f"ready artifact hash mismatch: {stage}/{name}")
        print(json.dumps({"stage": stage, "ready": str(dest / "plan_ready.json"),
                          "ready_sha256": sha(dest / "plan_ready.json"),
                          "plans": {layout: sha(dest / f"formal_{layout}_plan.json") for layout in LAYOUTS},
                          "fixed_denominator": expected}, indent=2))


if __name__ == "__main__":
    main()
