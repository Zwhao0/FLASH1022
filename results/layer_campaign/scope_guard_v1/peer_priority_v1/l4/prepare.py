#!/usr/bin/env python3
"""CPU-only L4 ATen-operator campaign plan and freeze helper.

Importing this module and running ``--check`` never draws a seed or writes a
design. ``make-timeonly`` freezes a fresh design and both schedules. The
separate ``authorize-timeonly`` command requires the completed upper6 gate;
``finalize`` requires stable health and valid physical FMA execution.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import math
import json
import random
import secrets
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
PEER_DIR = HERE.parent
SOURCE_FREEZE_PATH = PEER_DIR / "source_freeze.json"
UPPER_PREPARE_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/upper6/prepare.py"
L34_PREPARE_PATH = ROOT / "results/layer_campaign/l34_prepare.py"
L34_FAMILIES_PATH = ROOT / "results/layer_campaign/l34_family_sampling_draft.py"
BLIND8_PREPARE_PATH = PEER_DIR / "blind8/prepare.py"
MODEL_PATH = "results/v2_semantic/linear_candidate_model.json"
AUX_MODEL_PATH = "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped.json"
AUX_BINDING_PATH = "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped_context_binding.json"
LAYOUTS = ("legacy", "node_local_pp")
RANKS = tuple(range(16))
QUOTAS = {"matrix_multiply": 4, "elementwise_activation": 4, "routing_index": 3,
          "reduction_normalization": 2, "softmax": 1, "fused_sdpa_attention": 2}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


upper = _load("peer_v1_upper6_prepare_for_l4", UPPER_PREPARE_PATH)
l34 = _load("peer_v1_l34_prepare_for_l4", L34_PREPARE_PATH)
families = _load("peer_v1_l34_families_for_l4", L34_FAMILIES_PATH)
blind8 = _load("peer_v1_blind8_catalogs_for_l4", BLIND8_PREPARE_PATH)
# Reuse the peer-priority freeze while retaining the tested upper6 plan schema.
upper.SOURCE_FREEZE_PATH = SOURCE_FREEZE_PATH


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def source_catalogs_for_layout(layout):
    freeze, _, _ = upper.verify_freeze()
    amendment = blind8._verify_catalog_lineage(freeze)
    catalogs, refs = blind8._catalogs_for_layout(layout, freeze, amendment)
    return freeze, amendment, catalogs, refs


def filtered_fwd_operator_catalog(catalog):
    """Keep only declared families on framed, fully timed, positive-work sites."""
    selected, excluded = [], {}
    for row in catalog.get("operators", []):
        valid, detail = blind8._is_nonzero_framed_fwd_operator(row)
        if not valid:
            reason = str(detail)
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        try:
            family = families.operator_family(row)
        except ValueError:
            excluded["unmapped_overload"] = excluded.get("unmapped_overload", 0) + 1
            continue
        if family not in QUOTAS:
            excluded[f"recognized_but_unallocated:{family}"] = excluded.get(
                f"recognized_but_unallocated:{family}", 0) + 1
            continue
        selected.append(row)
    filtered = dict(catalog)
    filtered["operators"] = selected
    return filtered, excluded


def check_catalogs():
    old_design = read(ROOT / upper.base.OLD_DESIGN_PATH)
    attestation, summary = upper.base.attestation_inputs(old_design)
    if summary.get("localizer_outputs_used") is not False:
        raise ValueError("catalog attestation must not use localization results")
    freeze, _, _, _ = source_catalogs_for_layout("legacy")
    results = {"status": "candidate_catalogs_audited", "attestation": {
        "path": attestation["path"], "sha256": attestation["sha256"],
        "localizer_outputs_used": summary["localizer_outputs_used"]},
        "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
        "scope_catalog_amendment_sha256": sha(ROOT / "results/layer_campaign/scope_guard_v1/catalog_source_amendment.json"),
        "layouts": {}}
    for layout in LAYOUTS:
        _, _, raw_catalogs, catalog_refs = source_catalogs_for_layout(layout)
        catalogs, populations_by_rank = {}, {}
        population_counts = {family: 0 for family in QUOTAS}
        rank_coverage = {family: 0 for family in QUOTAS}
        dose_ratios = {family: [] for family in QUOTAS}
        unframed_functional_excluded = 0
        signature_or_timing_excluded = {}
        unmapped_operator_rows = 0
        baseline = upper.base.old_whole_baseline(old_design, layout)
        for rank in RANKS:
            catalog = raw_catalogs[rank]
            reference = catalog_refs[rank]
            if (reference.get("catalog_path") is None or reference.get("proof_sha256") is None
                    or catalog.get("component_operator_phases") != ["fwd"]):
                raise ValueError(f"missing catalog/proof or non-FWD operator population: {layout} rank {rank}")
            coverage = catalog.get("phase_coverage", {})
            if any(int(coverage.get(phase, {}).get("compute_calls", 0)) <= 0
                   for phase in ("fwd", "bwd")):
                raise ValueError(f"incomplete stage scope proof: {layout} rank {rank}")
            unframed_functional_excluded += sum(
                row.get("module_path") == "<training_functional>"
                or str(row.get("slot_id", "")).startswith("<training_functional>|")
                for row in catalog.get("operators", []))
            filtered, excluded = filtered_fwd_operator_catalog(catalog)
            catalogs[rank] = filtered
            unmapped_operator_rows += excluded.get("unmapped_overload", 0)
            for reason, count in excluded.items():
                if reason != "unmapped_overload":
                    signature_or_timing_excluded[reason] = signature_or_timing_excluded.get(reason, 0) + count
            populations = families.catalog_populations(filtered, "l4")
            populations_by_rank[rank] = populations
            for family in QUOTAS:
                rows = populations.get(family, [])
                if rows:
                    rank_coverage[family] += 1
                population_counts[family] += len(rows)
                for site in rows:
                    if (not site.get("slot_id") or site.get("phase") != "fwd"
                            or not site.get("operator") or int(site.get("calls", 0)) <= 0
                            or int(site.get("timing_unresolved_calls", 0)) != 0):
                        raise ValueError(f"invalid/unresolved L4 FWD operator site: {layout} rank {rank} {site.get('operator')}")
                    estimate = families.dose_from_catalog_site(site, baseline, 0.80, reference_steps=3)
                    dose_ratios[family].append(estimate["slowdown_extra_ratio"])
        if not l34._quota_match_feasible(populations_by_rank, QUOTAS):
            raise ValueError(f"L4 family quota is not rank-feasible: {layout}")
        calibration_units, bucket_ms = [], []
        for rank in RANKS:
            units, max_median_ms, _ = l34._fma_calibration(layout, rank)
            calibration_units.append(units)
            bucket_ms.append(max_median_ms)
        max_round_bound = int(math.ceil(
            (3.0 * baseline * 0.80 + max(bucket_ms)) * max(calibration_units))) + 1
        if max_round_bound >= l34.FMA_ROUNDS_MAX:
            raise ValueError(f"L4 conservative FMA round bound exceeds int limit: {layout}")
        results["layouts"][layout] = {
            "rank_count": 16, "family_population_sites": population_counts,
            "unframed_training_functional_sites_excluded": unframed_functional_excluded,
            "excluded_by_signature_timing_or_family": signature_or_timing_excluded,
            "excluded_unmapped_overload_sites": unmapped_operator_rows,
            "family_rank_coverage": rank_coverage, "rank_quota_match_feasible": True,
            "catalog_hashes": {str(rank): catalog_refs[rank]["catalog_sha256"] for rank in RANKS},
            "dose_reference_steps": 3,
            "max_slowdown_ratio_at_80pct_by_family": {name: max(values) for name, values in dose_ratios.items()},
            "conservative_fma_round_bound_at_80pct": max_round_bound,
        }
    return results


def check_schedules():
    dummy = [{"case_id": f"l4-case-{rank}", "rank": rank, "phase": "fwd",
              "fault_target": {"kind": "operator", "phase": "fwd", "slot_id": f"site-{rank}"},
              "dose": {"slowdown_extra_ratio": 1.0}} for rank in RANKS]
    timeonly, time_steps = upper.intervals_for("l2", dummy, "timeonly")
    formal, formal_steps = upper.intervals_for("l2", dummy, "formal")
    if time_steps != 207 or formal_steps != 1049:
        raise AssertionError((time_steps, formal_steps))
    for intervals, expected_steps, expected_fault in ((timeonly, 207, 5), (formal, 1049, 32)):
        faults = [row for row in intervals if row.get("segment") == "fault"]
        if len(faults) != 16 or intervals[-1]["end"] != expected_steps:
            raise AssertionError("L4 interval count or endpoint mismatch")
        if any(row["end"] - row["start"] + 1 != expected_fault for row in faults):
            raise AssertionError("L4 fault duration mismatch")
        if len({row["rank"] for row in faults}) != 16:
            raise AssertionError("L4 must have exactly one case per rank")
    return {"timeonly_steps_per_layout": time_steps, "formal_steps_per_layout": formal_steps,
            "cases_per_layout": 16, "operator_phase": "fwd"}


def check():
    report = check_catalogs()
    if sum(QUOTAS.values()) != 16 or QUOTAS != families.REVIEW_CANDIDATE_QUOTAS["l4"]:
        raise ValueError("L4 family quotas differ from the reviewed proposal")
    return {"passed": True, "writes_artifacts": False, "draws_seed_or_design": False,
            "fixed_denominator": 32, "quota_per_layout": QUOTAS,
            "scope_limit": "framed module-local allowlisted ATen overload sites are FWD-only; unframed training_functional sites excluded; no BWD hooks or arbitrary CUDA-kernel identity",
            "schedule": check_schedules(), "attestation": report["attestation"],
            "layouts": report["layouts"],
            "operator_semantics": {"softmax": ["aten._softmax.default"],
                                   "fused_sdpa_attention": ["aten._scaled_dot_product_flash_attention.default"],
                                   "sdpa_is_softmax": False,
                                   "unframed_training_functional_excluded": True,
                                   "log_softmax_loss_sampled": False,
                                   "site_identity": "preserve exact ATen overload and catalog slot_id"}}


def require_upper6_pass():
    checks = {}
    for stage, denominator in (("l1", 2), ("l2", 4)):
        path = PEER_DIR / "upper6" / stage / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"L4 time-only authorization requires completed upper6 {stage.upper()}")
        summary = read(path)
        by_layout = summary.get("by_layout")
        if not isinstance(by_layout, dict) or set(by_layout) != set(LAYOUTS):
            raise RuntimeError(f"upper6 {stage.upper()} summary must contain both layout health counts")
        healthy_by_layout = {layout: by_layout[layout].get("healthy_localized_false_alarms")
                             if isinstance(by_layout[layout], dict) else None
                             for layout in LAYOUTS}
        if any(type(value) is not int for value in healthy_by_layout.values()):
            raise RuntimeError(f"upper6 {stage.upper()} layout health counts must be integer values")
        healthy_wrong = sum(healthy_by_layout.values())
        if (summary.get("fixed_denominator") != denominator
                or summary.get("strict_success") != denominator
                or summary.get("gate_passed") is not True
                or healthy_wrong != 0):
            raise RuntimeError(f"upper6 {stage.upper()} must pass {denominator}/{denominator} with zero healthy wrong-GPU localizations")
        checks[stage] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                         "strict_success": denominator, "fixed_denominator": denominator,
                         "healthy_localized_false_alarms": healthy_wrong,
                         "healthy_localized_false_alarms_by_layout": healthy_by_layout}
    return {"passed": True, "combined_strict_success": 6, "combined_fixed_denominator": 6,
            "healthy_localized_false_alarms": 0, "stages": checks}


def make_cases(layout, rng, old_design, attestation):
    _, _, raw_catalogs, catalog_refs = source_catalogs_for_layout(layout)
    catalogs, excluded = {}, {}
    for rank in RANKS:
        catalogs[rank], excluded[rank] = filtered_fwd_operator_catalog(raw_catalogs[rank])
    sampled = families.sample_layout(catalogs, "l4", rng, QUOTAS)
    baseline = upper.base.old_whole_baseline(old_design, layout)
    topology_ref, topology = upper.base.topology_by_rank(old_design, layout)
    cases = []
    for sample in sampled:
        rank, family, site = sample["rank"], sample["family"], sample["site"]
        dose = families.dose_from_catalog_site(site, baseline, rng.uniform(0.40, 0.80), reference_steps=3)
        case_id = f"l4-{layout}-{family}-{rank}-{rng.getrandbits(32):08x}"
        cases.append({"case_id": case_id, "layout": layout, "layer": "L4",
                      "category": family, "family": family, "rank": rank,
                      "phase": "fwd", "population_size": sample["population_size"],
                      "sample_basis": sample["sample_basis"], "site": site,
                      "scope_catalog_sha256": catalogs[rank]["catalog_sha256"],
                      "scope_catalog_ref": catalog_refs[rank],
                      "nonzero_work_filter": {"all_tensor_shapes_positive": True,
                                               "predicate_source": "peer_priority_v1/blind8/prepare.py:_is_nonzero_framed_fwd_operator"},
                      "operator": site["operator"],
                      "fault_target": {"kind": "operator", "phase": "fwd", "slot_id": site["slot_id"]},
                      "dose": dose, "training_seed": None, "topology": topology[rank],
                      "topology_source": {"path": topology_ref["path"], "sha256": topology_ref["sha256"]}})
    return cases


def make_plan(layout, mode, cases, design_sha, source_sha, design, model):
    # Reuse the upper6 runner contract, specializing only L4 identity/scope.
    plan = upper.make_plan("l2", layout, mode, cases, design_sha, source_sha, design, model)
    plan.update({"protocol": f"peer_v1_l4_{mode}_v1",
                 "session_id": f"v2s_peer_l4_{mode}_{layout}_01",
                 "layer": "L4", "role": f"peer_l4_{mode}",
                 "fault_kind": "operator_compute", "fixed_denominator": 32,
                 "max_profiles": 40 if mode == "formal" else 0,
                 "operator_population_phase": "fwd",
                 "operator_family_quotas_per_layout": QUOTAS,
                 "candidate_population_note": "one eligible, healthy-executed FWD ATen overload slot per rank; exact overload and slot_id retained; family-stratified quota; not a CUDA-kernel identity claim"})
    if mode == "formal":
        plan["stage_forward_aux_enabled"] = True
    return plan


def make_timeonly():
    out = HERE
    generated = ("design.json", "timeonly_legacy_plan.json", "timeonly_node_local_pp_plan.json",
                 "candidate_audit.json", "design_lock.json", "timeonly_ready.json",
                 "formal_legacy_plan.json", "formal_node_local_pp_plan.json", "plan_ready.json")
    present = [name for name in generated if (out / name).exists()]
    if present:
        raise FileExistsError(f"preserve existing L4 artifacts: {present}")
    out.mkdir(parents=True, exist_ok=True)
    catalog_audit = check()
    source_freeze, peer_model, _ = upper.verify_freeze()
    if source_freeze.get("model_sha256") != read(ROOT / MODEL_PATH).get("sha256"):
        raise ValueError("peer source freeze must retain dc89 as the core model")
    old_design = read(ROOT / upper.base.OLD_DESIGN_PATH)
    attestation, attestation_summary = upper.base.attestation_inputs(old_design)
    if attestation.get("localizer_outputs_used") is not False:
        raise ValueError("catalog population must not use localization results")

    design_seed_hex = secrets.token_hex(16)
    rng = random.Random(int(design_seed_hex, 16))
    training_seed = rng.randrange(1, 2**31 - 1)
    audit_path = out / "candidate_audit.json"
    atomic_json(audit_path, catalog_audit)
    design = {"protocol": "peer_priority_v1_l4_design_v1", "layer": "L4",
              "integration_not_blind": False, "independent_randomized_acceptance": True,
              "prediction_before_truth": True, "design_seed_hex": design_seed_hex,
              "training_seed": training_seed, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_freeze": {"path": str(SOURCE_FREEZE_PATH.relative_to(ROOT)),
                                "sha256": sha(SOURCE_FREEZE_PATH),
                                "dc89_sha256": source_freeze["model_sha256"]},
              "plan_builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                               "sha256": sha(Path(__file__).resolve())},
              "candidate_catalog_audit": {"path": str(audit_path.relative_to(ROOT)),
                  "sha256": sha(audit_path), "attestation_sha256": catalog_audit["attestation"]["sha256"]},
              "execution_prerequisite": {"upper6_must_pass_before_timeonly_authorization": True,
                  "required_strict_success": 6, "fixed_denominator": 6,
                  "healthy_localized_false_alarms": 0,
                  "upper6_results_are_not_design_inputs": True},
              "acceptance_gate": {"strict_success_threshold": 26, "fixed_denominator": 32},
              "fixed_denominator": 32, "cases_per_layout": 16,
              "operator_family_quotas_per_layout": QUOTAS,
              "operator_population_phase": "fwd",
              "operator_family_semantics": {"matrix_multiply": ["aten.mm.default", "aten.addmm.default", "aten.bmm.default"],
                  "elementwise_activation": ["aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor", "aten.neg.default", "aten.pow.Tensor_Scalar", "aten.silu.default", "aten.sigmoid.default"],
                  "routing_index": ["aten.eq.Scalar", "aten.ge.Scalar", "aten.lt.Scalar", "aten.index_add.default", "aten.topk.default", "aten.sort.default", "aten.sort.stable", "aten.masked_fill.Scalar"],
                  "reduction_normalization": ["aten.sum.default", "aten.sum.dim_IntList", "aten.mean.dim", "aten.rsqrt.default"],
                  "softmax": ["aten._softmax.default"],
                  "fused_sdpa_attention": ["aten._scaled_dot_product_flash_attention.default"]},
              "unallocated_recognized_operator_family": {"log_softmax_loss": ["aten._log_softmax.default", "aten.nll_loss_forward.default"]},
              "sdpa_is_softmax": False,
              "scope_limit": "framed FWD only; unframed training_functional excluded; actual allowlisted ATen overload sites, not arbitrary CUDA-kernel identity",
              "phase": "fwd", "dose_policy": "40-80% healthy wall target; three-step catalog total divided by 3; one time-only confirmation; no replacement or retuning",
              "schedule": {"timeonly_steps_per_layout": 207, "formal_steps_per_layout": 1049,
                           "timeonly": "12 warmup + 30 stable-health gate + 16*(5 healthy + 5 fault) + 5 tail",
                           "formal": "42 healthy opening + 16*(32 fault + 30 healthy) + 15 tail"},
              "layouts": {}}
    design_path = out / "design.json"
    for layout in LAYOUTS:
        design["layouts"][layout] = make_cases(layout, rng, old_design, attestation)
        for case in design["layouts"][layout]:
            case["training_seed"] = training_seed
        counts = {family: sum(row["family"] == family for row in design["layouts"][layout]) for family in QUOTAS}
        if counts != QUOTAS:
            raise AssertionError(f"L4 sampled operator family quotas differ for {layout}: {counts}")
    atomic_json(design_path, design)
    design_sha, source_sha = sha(design_path), sha(SOURCE_FREEZE_PATH)
    for layout in LAYOUTS:
        plan = make_plan(layout, "timeonly", design["layouts"][layout], design_sha,
                         source_sha, design, peer_model)
        atomic_json(out / f"timeonly_{layout}_plan.json", plan)
        formal_plan = make_plan(layout, "formal", design["layouts"][layout], design_sha,
                                source_sha, design, peer_model)
        atomic_json(out / f"formal_{layout}_plan.json", formal_plan)
    names = ["candidate_audit.json", "design.json", "timeonly_legacy_plan.json",
             "timeonly_node_local_pp_plan.json", "formal_legacy_plan.json", "formal_node_local_pp_plan.json"]
    design_lock = {"status": "design_and_both_schedules_locked_no_gpu_ready", "stage": "L4",
                   "fixed_denominator": 32, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_freeze_sha256": source_sha, "design_sha256": design_sha,
                   "timeonly_steps_per_layout": 207, "formal_steps_per_layout": 1049,
                   "gpu_authorized": False,
                   "artifacts": {name: sha(out / name) for name in names}}
    atomic_json(out / "design_lock.json", design_lock)
    return design_lock


def authorize_timeonly():
    lock_path = HERE / "design_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("run make-timeonly once to lock L4 design and schedules")
    lock = read(lock_path)
    if lock.get("gpu_authorized") is not False or lock.get("fixed_denominator") != 32:
        raise ValueError("invalid L4 design lock")
    if sha(SOURCE_FREEZE_PATH) != lock.get("source_freeze_sha256"):
        raise ValueError("L4 source freeze changed after design draw")
    for name, digest in lock.get("artifacts", {}).items():
        if not (HERE / name).is_file() or sha(HERE / name) != digest:
            raise ValueError(f"L4 frozen design artifact changed: {name}")
    upper6_gate = require_upper6_pass()
    ready_path = HERE / "timeonly_ready.json"
    if ready_path.exists():
        raise FileExistsError("preserve existing L4 timeonly ready marker")
    names = ["source_freeze.json", "design_lock.json", *lock["artifacts"].keys()]
    ready = {"status": "timeonly_ready_after_upper6_gate", "stage": "L4",
             "fixed_denominator": 32, "created_utc": datetime.now(timezone.utc).isoformat(),
             "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
             "design_sha256": sha(HERE / "design.json"), "prior_upper6_gate": upper6_gate,
             "artifacts": {name: sha(SOURCE_FREEZE_PATH if name == "source_freeze.json" else HERE / name)
                           for name in names}}
    atomic_json(ready_path, ready)
    return ready


def finalize():
    out = HERE
    recovery_ready_path = out / "recovery_ready.json"
    if recovery_ready_path.is_file():
        recovery = read(recovery_ready_path)
        amendment_path = out / "recovery_amendment.json"
        gate_path = out / "timeonly_gate.json"
        if (recovery.get("status") != "recovery_ready_for_single_gpu_owner"
                or recovery.get("fixed_denominator") != 32
                or recovery.get("recovery_amendment_sha256") != sha(amendment_path)):
            raise RuntimeError("L4 recovery-ready/amendment binding is invalid")
        for name, digest in recovery.get("artifacts", {}).items():
            if name == "cleanup.json":
                continue
            path = SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name
            if not path.is_file() or sha(path) != digest:
                raise RuntimeError(f"L4 recovery input changed: {name}")
        if not gate_path.is_file():
            raise FileNotFoundError("L4 recovery time-only gate is required before formal readiness")
        gate = read(gate_path)
        if (gate.get("protocol") != "peer_v1_l4_recovery_timeonly_gate_v1"
                or gate.get("recovery_ready_sha256") != sha(recovery_ready_path)
                or gate.get("recovery_amendment_sha256") != sha(amendment_path)
                or gate.get("stable_for_formal") is not True
                or gate.get("fixed_denominator") != 32
                or gate.get("source_freeze_sha256") != sha(SOURCE_FREEZE_PATH)
                or gate.get("design_sha256") != sha(out / "design.json")):
            raise RuntimeError("L4 recovery health/FMA gate or its lineage did not pass")
        for layout in LAYOUTS:
            expected = recovery.get("effective_plans", {}).get("timeonly", {}).get(layout)
            plan_name = f"recovery_timeonly_{layout}_plan.json"
            if expected != plan_name:
                raise RuntimeError(f"L4 recovery time-only selector mismatch: {layout}")
            if gate.get("effective_plan_sha256", {}).get(layout) != sha(out / plan_name):
                raise RuntimeError(f"L4 recovery gate did not use the selected plan: {layout}")
        if require_upper6_pass() != recovery.get("prior_upper6_gate"):
            raise RuntimeError("upper6 pass binding changed after L4 recovery authorization")
        names = ["source_freeze.json", "design_lock.json", "timeonly_ready.json",
                 "recovery_ready.json", "recovery_amendment.json", "actual_catalog_loader_smoke.json",
                 "design.json", "candidate_audit.json", "timeonly_gate.json",
                 "recovery_timeonly_legacy_plan.json", "recovery_timeonly_node_local_pp_plan.json",
                 "recovery_formal_legacy_plan.json", "recovery_formal_node_local_pp_plan.json"]
        artifacts = {name: sha(SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name)
                     for name in names}
        ready = {"status": "ready_for_single_gpu_owner", "stage": "L4",
                 "mode": "metadata_only_recovery", "created_utc": datetime.now(timezone.utc).isoformat(),
                 "fixed_denominator": 32, "source_freeze_sha256": sha(SOURCE_FREEZE_PATH),
                 "design_sha256": sha(out / "design.json"),
                 "timeonly_gate_sha256": sha(gate_path),
                 "recovery_ready_sha256": sha(recovery_ready_path),
                 "recovery_amendment_sha256": sha(amendment_path),
                 "upper6_gate_sha256": canonical_hash(recovery.get("prior_upper6_gate")),
                 "effective_plans": {"formal": {layout: f"recovery_formal_{layout}_plan.json"
                                                  for layout in LAYOUTS}},
                 "artifacts": artifacts}
        atomic_json(out / "plan_ready.json", ready)
        return ready
    ready_path = out / "timeonly_ready.json"
    gate_path = out / "timeonly_gate.json"
    if not ready_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError("L4 upper6-authorized timeonly_ready and completed timeonly gate are required")
    ready0, gate = read(ready_path), read(gate_path)
    if (ready0.get("status") != "timeonly_ready_after_upper6_gate"
            or ready0.get("fixed_denominator") != 32
            or gate.get("stable_for_formal") is not True
            or gate.get("fma_structure_passed") is not True
            or gate.get("fixed_denominator") != 32
            or gate.get("timeonly_ready_sha256") != sha(ready_path)):
        raise RuntimeError("L4 formal plan remains locked until health stability and FMA structure gates pass")
    if sha(SOURCE_FREEZE_PATH) != ready0["source_freeze_sha256"]:
        raise RuntimeError("peer source freeze changed after L4 draw")
    for name, digest in ready0["artifacts"].items():
        path = SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name
        if sha(path) != digest:
            raise RuntimeError(f"L4 time-only input changed: {name}")
    if require_upper6_pass() != ready0.get("prior_upper6_gate"):
        raise RuntimeError("upper6 pass binding changed after L4 design authorization")
    names = ["source_freeze.json", "timeonly_gate.json", "design_lock.json", "candidate_audit.json",
             "design.json", "timeonly_ready.json",
             "timeonly_legacy_plan.json", "timeonly_node_local_pp_plan.json",
             "formal_legacy_plan.json", "formal_node_local_pp_plan.json"]
    artifacts = {name: sha(SOURCE_FREEZE_PATH if name == "source_freeze.json" else out / name) for name in names}
    ready = {"status": "ready_for_single_gpu_owner", "stage": "L4",
             "created_utc": datetime.now(timezone.utc).isoformat(), "fixed_denominator": 32,
             "source_freeze_sha256": sha(SOURCE_FREEZE_PATH), "design_sha256": sha(out / "design.json"),
             "timeonly_gate_sha256": sha(gate_path),
             "upper6_gate_sha256": canonical_hash(ready0.get("prior_upper6_gate")),
             "artifacts": artifacts}
    atomic_json(out / "plan_ready.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs="?", choices=("make-timeonly", "authorize-timeonly", "finalize"))
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.command == "make-timeonly":
        value = make_timeonly()
    elif args.command == "authorize-timeonly":
        value = authorize_timeonly()
    elif args.command == "finalize":
        value = finalize()
    else:
        parser.error("choose --check, make-timeonly, authorize-timeonly, or finalize")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
