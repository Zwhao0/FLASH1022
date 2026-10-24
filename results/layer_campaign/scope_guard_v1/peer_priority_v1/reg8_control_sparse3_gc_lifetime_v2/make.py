#!/usr/bin/env python3
"""Freeze the unchanged fixed-reg8 cases against sparse3+GC staging.

This CPU-only maker copies the fixed eight intervals from the prior sparse4
candidate, adds only the sparse3 sampling and opt-in diagnostic flags, and
constructs an isolated staged source tree. It never starts a training run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_sparse4_gc_lifetime_v2"
BASE_PROD = ROOT / "results/layer_campaign/stage_forward_v1/sparse3_gc_production_draft"
DIAG = ROOT / "results/layer_campaign/stage_forward_v1/sparse3_gc_diag_draft"
PREVIOUS = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_reverted_baseline"
LAYOUTS = ("legacy", "node_local_pp")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def build_stage():
    base_tree = read(BASE_PROD / "staged_tree.json")
    base_tar = BASE_PROD / "staged_flash_v1.tar"
    diag_ready = read(DIAG / "ready.json")
    if sha(base_tar) != base_tree["tar_sha256"]:
        raise ValueError("sparse3 production tar differs from frozen staged tree")
    if (diag_ready.get("status") != "CPU_optional_diagnostic_overlay_not_applied"
            or diag_ready.get("default_disabled") is not True
            or diag_ready.get("plan_flags") != {"gc_callbacks_diagnostic": True, "monitor_cost_timing": True}
            or diag_ready.get("base_source_binding_sha256") != sha(BASE_PROD / "source_binding.json")):
        raise ValueError("GC diagnostic overlay is not pinned to the sparse3 production base")

    stage_root = HERE / "production"
    flash = stage_root / "flash_v1"
    if flash.exists():
        raise FileExistsError("candidate staged source already exists; preserve it")
    stage_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(base_tar, "r") as archive:
        members = archive.getmembers()
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError(f"unsafe/non-file base member: {member.name}")
        archive.extractall(stage_root)

    for rel, record in diag_ready["files"].items():
        destination = stage_root / rel
        expected_before = record.get("before_sha256")
        actual_before = sha(destination) if destination.is_file() else None
        if actual_before != expected_before:
            raise ValueError(f"diagnostic overlay base mismatch: {rel}")
        overlay = DIAG / "overlay" / rel
        if not overlay.is_file() or sha(overlay) != record["after_sha256"]:
            raise ValueError(f"diagnostic overlay artifact mismatch: {rel}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay, destination)

    tree_files = {}
    tar_path = HERE / "staged_flash_v1.tar"
    with tarfile.open(tar_path, "w") as archive:
        for path in sorted(p for p in flash.rglob("*") if p.is_file()):
            rel = path.relative_to(stage_root).as_posix()
            tree_files[rel] = sha(path)
            info = tarfile.TarInfo(rel)
            info.size = path.stat().st_size
            info.mode = path.stat().st_mode & 0o777
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    staged_tree = dict(
        schema="sparse3_gcdiag_staged_flash_tree_v1",
        parent_tree_sha256=sha(BASE_PROD / "staged_tree.json"),
        parent_tar_sha256=sha(base_tar),
        diagnostic_ready_sha256=sha(DIAG / "ready.json"),
        diagnostic_patch_sha256=diag_ready["patch_sha256"],
        files=tree_files,
        tar_sha256=sha(tar_path),
    )
    write(HERE / "staged_tree.json", staged_tree)
    return staged_tree, diag_ready


def main():
    if any((HERE / name).exists() for name in (
        "design.json", "source_freeze.json", "ready.json", "formal_legacy_plan.json",
        "formal_node_local_pp_plan.json", "execution_start.json", "execution_complete.json",
    )):
        raise FileExistsError("reg8 artifacts already exist; preserve the frozen version")
    old_source = read(OLD / "source_freeze.json")
    old_design = read(OLD / "design.json")
    base_source = read(BASE_PROD / "source_binding.json")
    base_production_freeze = read(BASE_PROD / "freeze.json")
    diag_ready = read(DIAG / "ready.json")
    staged_tree, diag_ready = build_stage()

    # Start from the already source-bound fixed-case catalog and update every
    # flash_v1 digest to the actual sparse3+diagnostic tree we deploy.
    files = dict(old_source["files"])
    files.update(staged_tree["files"])
    freeze = dict(old_source)
    freeze.update(
        protocol="peer_priority_reg8_sparse3_startup_gc_gcdiag_v1",
        status="frozen_isolated_candidate_not_live",
        files=dict(sorted(files.items())),
        collector_sha256="57c7cd2d49e231448354521d51eaeb12d078b7e242ddc22f34c088fd5766a204",
        parent_source_freeze_path=str((OLD / "source_freeze.json").relative_to(ROOT)),
        parent_source_freeze_sha256=sha(OLD / "source_freeze.json"),
        staged_production_patch_sha256=base_source["patch_sha256"],
        staged_production_source_binding_path=str((BASE_PROD / "source_binding.json").relative_to(ROOT)),
        staged_production_source_binding_sha256=sha(BASE_PROD / "source_binding.json"),
        staged_production_freeze_path=str((BASE_PROD / "freeze.json").relative_to(ROOT)),
        staged_production_freeze_sha256=sha(BASE_PROD / "freeze.json"),
        gc_diagnostic_overlay_path=str((DIAG / "ready.json").relative_to(ROOT)),
        gc_diagnostic_overlay_sha256=sha(DIAG / "ready.json"),
        gc_diagnostic_patch_sha256=diag_ready["patch_sha256"],
        staged_tree_path=str((HERE / "staged_tree.json").relative_to(ROOT)),
        staged_tree_sha256=sha(HERE / "staged_tree.json"),
        staged_tar_path=str((HERE / "staged_flash_v1.tar").relative_to(ROOT)),
        staged_tar_sha256=sha(HERE / "staged_flash_v1.tar"),
        sampling_schema="flash_stage_forward_sparse3_v1",
        sample_offsets=[1, 3, 7],
        expected_aux_rows_per_window=48,
        gc_callbacks_diagnostic=True,
        monitor_cost_timing=True,
        no_live_source_change=True,
    )
    freeze_path = HERE / "source_freeze.json"
    write(freeze_path, freeze)

    sampling_path = HERE / "production/flash_v1/stage_forward_sampling.py"
    sampling_ns = {}
    exec(compile(sampling_path.read_text(), str(sampling_path), "exec"), sampling_ns)
    sampling = sampling_ns["CONFIG"]
    plans = {}
    plan_amendments = {}
    old_plans = {}
    for layout in LAYOUTS:
        old_path = OLD / f"formal_{layout}_plan.json"
        plan = read(old_path)
        parent_plan = copy.deepcopy(plan)
        plan["session_id"] = f"v2s_peer_priority_reg8_sparse3_gc_diag_{layout}_01"
        plan["freeze_manifest_path"] = str(freeze_path.relative_to(ROOT))
        plan["freeze_manifest_sha256"] = sha(freeze_path)
        plan["stage_forward_aux_sampling"] = copy.deepcopy(sampling)
        plan["gc_callbacks_diagnostic"] = True
        plan["monitor_cost_timing"] = True
        if [r for r in plan["intervals"] if r.get("segment") == "fault"] != [
            r for r in parent_plan["intervals"] if r.get("segment") == "fault"
        ]:
            raise AssertionError("fault intervals changed during cloning")
        plan_path = HERE / f"formal_{layout}_plan.json"
        write(plan_path, plan)
        plans[layout] = plan
        old_plans[layout] = parent_plan
        plan_amendments[layout] = dict(
            path=str(plan_path.relative_to(ROOT)),
            sha256=sha(plan_path),
            parent_path=str(old_path.relative_to(ROOT)),
            parent_sha256=sha(old_path),
            changed_fields=sorted(k for k in set(plan) | set(parent_plan) if plan.get(k) != parent_plan.get(k)),
            session_id=plan["session_id"],
            same_intervals=plan["intervals"] == parent_plan["intervals"],
            same_seed=plan["seed"] == parent_plan["seed"],
        )
    amendment = dict(
        schema="sparse3_gcdiag_reg8_plan_amendment_v1",
        previous_candidate_path=str(OLD.relative_to(ROOT)),
        previous_candidate_source_freeze_sha256=sha(OLD / "source_freeze.json"),
        previous_case_design_sha256=sha(OLD / "design.json"),
        candidate_source_freeze_path=str(freeze_path.relative_to(ROOT)),
        candidate_source_freeze_sha256=sha(freeze_path),
        source_delta=dict(
            sparse3_production_freeze_sha256=sha(BASE_PROD / "freeze.json"),
            gc_diagnostic_overlay_sha256=sha(DIAG / "ready.json"),
            sampling=sampling,
            monitor_cost_timing=True,
            gc_callbacks_diagnostic=True,
        ),
        layouts=plan_amendments,
        intervals_seed_sites_strengths_unchanged=True,
        no_localization_in_design=True,
        no_live_source_change=True,
    )
    amendment_path = HERE / "source_plan_amendment.json"
    write(amendment_path, amendment)

    design = copy.deepcopy(old_design)
    design["protocol"] = "peer_priority_reg8_sparse3_startup_gc_gcdiag_fixed_case_regression_v1"
    design["source_freeze"] = dict(path=str(freeze_path.relative_to(ROOT)), sha256=sha(freeze_path))
    design["parent_design_path"] = str((OLD / "design.json").relative_to(ROOT))
    design["parent_design_sha256"] = sha(OLD / "design.json")
    design["layout_plan_amendments"] = {
        layout: dict(
            candidate_plan_path=plan_amendments[layout]["path"],
            candidate_plan_sha256=plan_amendments[layout]["sha256"],
            parent_plan_path=plan_amendments[layout]["parent_path"],
            parent_plan_sha256=plan_amendments[layout]["parent_sha256"],
            only_changed_fields=plan_amendments[layout]["changed_fields"],
            session_id=plan_amendments[layout]["session_id"],
        ) for layout in LAYOUTS
    }
    design["sampling_schema"] = sampling
    design["monitor_cost_timing"] = True
    design["gc_callbacks_diagnostic"] = True
    design_path = HERE / "design.json"
    write(design_path, design)

    contract = dict(
        schema="peer_priority_reg8_sparse3_gcdiag_contract_v1",
        fixed_denominator=8,
        steps_per_layout=305,
        seed=2032905414,
        source_freeze_path=str(freeze_path.relative_to(ROOT)),
        source_freeze_sha256=sha(freeze_path),
        design_path=str(design_path.relative_to(ROOT)),
        design_sha256=sha(design_path),
        parent_baseline_summary_path=str((PREVIOUS / "summary.json").relative_to(ROOT)),
        parent_baseline_summary_sha256=sha(PREVIOUS / "summary.json"),
        exact_case_intervals=True,
        stage_forward_aux_sampling=sampling,
        gc_callbacks_diagnostic=True,
        monitor_cost_timing=True,
        startup_gc_fixed_model=True,
        prediction_before_truth=True,
        no_automatic_retry=True,
        no_live_source_change=True,
        execution_order=list(LAYOUTS),
    )
    contract_path = HERE / "contract.json"
    write(contract_path, contract)

    artifacts = {
        path.name: sha(path)
        for path in (
            design_path, contract_path, freeze_path, amendment_path,
            HERE / "formal_legacy_plan.json", HERE / "formal_node_local_pp_plan.json",
            HERE / "staged_tree.json", HERE / "staged_flash_v1.tar",
        )
    }
    ready = dict(
        schema="peer_priority_reg8_sparse3_gcdiag_ready_v1",
        status="ready_for_new_fixed_case_sparse3_gcdiag_regression",
        fixed_denominator=8,
        steps_per_layout=305,
        local_live_source_applied=False,
        no_automatic_retry=True,
        prediction_before_truth=True,
        design_path=str(design_path.relative_to(ROOT)),
        design_sha256=sha(design_path),
        source_freeze_path=str(freeze_path.relative_to(ROOT)),
        source_freeze_sha256=sha(freeze_path),
        artifacts=artifacts,
        candidate_gate_rule="same fixed eight cases; preserve all seven prior baseline successes; all six whole/BWD/FWD-component cases strict-success; healthy wrong-GPU localizations zero; sparse3 auxiliary must be complete and valid on all evidence windows; no weak, invalid, rejected or missed case removed",
    )
    write(HERE / "ready.json", ready)
    print(json.dumps({"status": ready["status"], "ready_sha256": sha(HERE / "ready.json"),
                      "source_freeze_sha256": sha(freeze_path), "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
