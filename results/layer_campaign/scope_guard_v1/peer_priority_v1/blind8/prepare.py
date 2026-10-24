#!/usr/bin/env python3
"""CPU-only component blind8 population and schedule audit.

This helper audits candidate catalogs and the fixed schedule only.  It never
draws a seed, selects a site, or writes a design/plan.  Run with ``--check``;
``--write-audit`` additionally saves the deterministic census.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
GUARD = PARENT.parent
SOURCE_FREEZE = PARENT / "source_freeze.json"
CATALOG_AMENDMENT = GUARD / "catalog_source_amendment.json"
OLD_DESIGN = ROOT / "results/recent_three8/design.json"
MODEL_PATH = ROOT / "results/v2_semantic/linear_candidate_model.json"
AUX_MODEL_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped.json"
AUX_BINDING_PATH = ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/model_scoped_context_binding.json"
LAYOUTS = ("legacy", "node_local_pp")
RANKS = tuple(range(16))
FAMILIES = ("attention", "routed_expert_mlp", "moe_block_router", "normalization")
QUOTAS = {family: 1 for family in FAMILIES}
TIMEONLY_STEPS = 12 + 30 + 4 * (5 + 5) + 5
FORMAL_STEPS = 42 + 4 * (32 + 30) + 15
EXPECTED_SOURCE_SHA256 = "3eb4b97a56c125039e09f203636fbf25eddaa19b9abd84a4fe659423da717e61"
EXPECTED_CORE_SHA256 = "dc89b9f6b6d593c95757ad644fcf5055806724b22b693ef08542bed222bdd300"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FAMILY_HELPER = _load_module(
    "blind8_l34_family_sampler",
    ROOT / "results/layer_campaign/l34_family_sampling_draft.py",
)
UPPER_PREPARE = _load_module(
    "blind8_upper6_prepare",
    ROOT / "results/layer_campaign/stage_forward_v1/peer_v1/upper6/prepare.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_shapes(value):
    """Yield tensor shapes from a literal catalog signature, never eval code."""
    if isinstance(value, (tuple, list)):
        if (len(value) >= 2 and value[0] == "tensor"
                and isinstance(value[1], (tuple, list))):
            yield tuple(value[1])
            return
        for child in value:
            yield from _tensor_shapes(child)


def _has_nonzero_tensor_work(signature_strings):
    """Require a recorded tensor signature with a nonzero element count."""
    if not isinstance(signature_strings, list) or not signature_strings:
        return False, 0, 0
    valid_signatures = 0
    positive_signatures = 0
    for encoded in signature_strings:
        if not isinstance(encoded,str):continue
        try:parsed=ast.literal_eval(encoded)
        except (ValueError,SyntaxError):continue
        shapes=list(_tensor_shapes(parsed))
        if not shapes or not all(all(type(d) is int and d>=0 for d in shape) for shape in shapes):continue
        valid_signatures+=1
        # A nonempty weight alone cannot make an empty-token GEMM real work.
        # Scalar tensors have shape () and one element; every participating
        # tensor in at least one actual healthy signature must be nonempty.
        if all(math.prod(shape)>0 for shape in shapes):positive_signatures+=1
    return positive_signatures>0,valid_signatures,positive_signatures


def _assert_nonzero_signature_examples():
    """Guard the empty-token GEMM boundary used by component filtering."""
    def sig(*shapes):
        tensors = tuple(("tensor", shape, tuple(1 for _ in shape), "torch.bfloat16", "cuda:0")
                        for shape in shapes)
        return repr((tensors, ()))

    empty_token_gemm = sig((0, 896), (896, 64))
    positive_gemm = sig((16, 896), (896, 64))
    scalar = sig(())
    assert _has_nonzero_tensor_work([empty_token_gemm])[0] is False
    assert _has_nonzero_tensor_work([positive_gemm])[0] is True
    assert _has_nonzero_tensor_work([scalar])[0] is True
    return {"empty_token_gemm_with_nonzero_weight": False,
            "positive_matrix_multiply": True, "scalar_tensor": True}


def _is_nonzero_framed_fwd_operator(row):
    module_path = row.get("module_path")
    slot_id = row.get("slot_id")
    if row.get("phase") != "fwd" or not isinstance(module_path, str) or not module_path:
        return False, "not_fwd_or_unframed"
    if (module_path in ("<model>", "<training_functional>")
            or module_path.startswith("<")
            or (isinstance(slot_id, str) and slot_id.startswith("<training_functional>|"))):
        return False, "not_framed_module_site"
    if not isinstance(slot_id, str) or not slot_id or not row.get("operator", "").startswith("aten."):
        return False, "missing_named_aten_site"
    calls = row.get("calls")
    unresolved = row.get("timing_unresolved_calls")
    total_ms = row.get("estimated_healthy_total_ms")
    timing_calls = row.get("timing_calls")
    if (type(calls) is not int or calls <= 0 or unresolved != 0
            or not isinstance(total_ms, (int, float)) or isinstance(total_ms, bool)
            or not math.isfinite(float(total_ms)) or total_ms <= 0
            or not isinstance(timing_calls, dict) or not timing_calls):
        return False, "not_fully_timed_compute"
    has_work, valid_signatures, positive_signatures = _has_nonzero_tensor_work(row.get("signatures"))
    if not has_work:
        return False, "no_nonzero_tensor_signature"
    return True, {"valid_tensor_signatures": valid_signatures,
                  "nonzero_tensor_signatures": positive_signatures}


def _explicitly_excluded_component(row):
    path = str(row.get("path", ""))
    module_class = str(row.get("module_class", ""))
    return (path in {"<model>", "decoder", "decoder.layers"}
            or module_class in {"StageModel", "Qwen2MoeModel", "Qwen2MoeDecoderLayer",
                                "ModuleList", "ModuleDict", "Sequential"})


def _load_and_verify_freeze():
    if not SOURCE_FREEZE.is_file():
        raise FileNotFoundError(f"peer-priority source freeze is missing: {SOURCE_FREEZE}")
    freeze = read_json(SOURCE_FREEZE)
    if sha(SOURCE_FREEZE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("peer-priority source freeze differs from the reviewed source")
    if freeze.get("model_sha256") != EXPECTED_CORE_SHA256:
        raise ValueError("source freeze no longer binds the frozen dc89 core model")
    mismatches = []
    for relative, expected in freeze.get("files", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"source files do not match the peer-priority freeze: {mismatches[:8]}")
    core_model = read_json(MODEL_PATH)
    if (core_model.get("sha256") != EXPECTED_CORE_SHA256
            or sha(MODEL_PATH) != freeze.get("files", {}).get("results/v2_semantic/linear_candidate_model.json")):
        raise ValueError("core model file hash changed")
    peer_model = read_json(AUX_MODEL_PATH)
    if sha(AUX_MODEL_PATH) != freeze.get("models", {}).get("scoped"):
        raise ValueError("scoped peer model does not match source freeze")
    binding = read_json(AUX_BINDING_PATH)
    if (binding.get("model_sha256") != peer_model.get("sha256")
            or sha(AUX_BINDING_PATH) != freeze.get("files", {}).get(str(AUX_BINDING_PATH.relative_to(ROOT)))):
        raise ValueError("scoped peer model context binding mismatch")
    return freeze, peer_model, binding


def _verify_catalog_lineage(freeze):
    if not CATALOG_AMENDMENT.is_file():
        raise FileNotFoundError(CATALOG_AMENDMENT)
    amendment = read_json(CATALOG_AMENDMENT)
    guard_binding = freeze.get("scope_guard_amendment", {})
    if (sha(CATALOG_AMENDMENT) != guard_binding.get("sha256")
            or amendment.get("status") != "cpu_only_no_new_gpu"):
        raise ValueError("catalog CPU source amendment is not bound to current source freeze")
    if amendment.get("new_scoped_source_sha256") != freeze.get("files", {}).get("flash_v1/scoped_compute.py"):
        raise ValueError("catalog amendment scoped source hash mismatch")
    return amendment


def _catalogs_for_layout(layout, freeze, amendment):
    entries = {(row["layout"], int(row["rank"])): row for row in amendment.get("catalogs", [])}
    catalogs = {}
    census_rows = {}
    current_session = ROOT / "results/data16_sessions" / f"v2s_scope_guard_catalog_{layout}_01"
    for rank in RANKS:
        entry = entries.get((layout, rank))
        if entry is None:
            raise ValueError(f"missing derived catalog amendment: {layout} rank {rank}")
        derived_path = ROOT / entry["new_path"]
        old_path = ROOT / entry["old_path"]
        active_path = current_session / f"scope_catalog_rank{rank}.json"
        if (sha(derived_path) != entry.get("new_file_sha256")
                or sha(old_path) != entry.get("old_file_sha256")):
            raise ValueError(f"catalog file hash mismatch: {layout} rank {rank}")
        old, derived = read_json(old_path), read_json(derived_path)
        current = read_json(active_path)
        if current != derived:
            raise ValueError(f"active source-bound catalog differs from CPU-derived catalog: {layout} rank {rank}")
        expected_scoped = freeze["files"]["flash_v1/scoped_compute.py"]
        expected_whole = freeze["files"]["flash_v1/whole_compute.py"]
        if (derived.get("source_sha256", {}).get("scoped_compute.py") != expected_scoped
                or derived.get("source_sha256", {}).get("whole_compute.py") != expected_whole
                or derived.get("validation") != "gpu_numerical_and_actual_scope_coverage_passed"
                or derived.get("rank") != rank
                or derived.get("component_operator_phases") != ["fwd"]
                or set(derived.get("phases", [])) != {"fwd", "bwd"}):
            raise ValueError(f"catalog source/scope contract mismatch: {layout} rank {rank}")
        old_copy, derived_copy = copy.deepcopy(old), copy.deepcopy(derived)
        old_copy.get("source_sha256", {}).pop("scoped_compute.py", None)
        derived_copy.get("source_sha256", {}).pop("scoped_compute.py", None)
        old_copy.pop("catalog_sha256", None)
        derived_copy.pop("catalog_sha256", None)
        if old_copy != derived_copy:
            raise ValueError(f"catalog population/trace changed beyond scoped source/hash: {layout} rank {rank}")
        if not entry.get("all_other_fields_identical") or not entry.get("new_strict_loader_passed"):
            raise ValueError(f"catalog amendment lacks its CPU deep-diff/loader attestation: {layout} rank {rank}")
        catalogs[rank] = derived
        census_rows[rank] = {"catalog_path": str(active_path.relative_to(ROOT)),
                             "catalog_sha256": sha(active_path),
                             "proof_sha256": derived.get("validation_proof_sha256"),
                             "derived_from": entry["old_path"]}
    return catalogs, census_rows


def _audit_components(catalogs, baseline):
    raw_counts = Counter()
    framed_work_counts = Counter()
    work_operator_counts = Counter()
    ranks_by_family = defaultdict(set)
    site_examples = defaultdict(list)
    unresolved_shapes = Counter()
    site_dose_ratios = defaultdict(list)
    for rank, catalog in catalogs.items():
        valid_ops = []
        for op in catalog.get("operators", []):
            valid, detail = _is_nonzero_framed_fwd_operator(op)
            if valid:
                valid_ops.append(op)
            elif detail in ("no_nonzero_tensor_signature", "not_fully_timed_compute"):
                unresolved_shapes[detail] += 1
        for component in catalog.get("components", []):
            family = FAMILY_HELPER.component_family(component)
            if family not in FAMILIES or _explicitly_excluded_component(component):
                continue
            path = component.get("path")
            slot = component.get("slot_id")
            if not component.get("eligible") or not isinstance(path, str) or not path or slot != path:
                continue
            raw_counts[family] += 1
            descendants = [op for op in valid_ops
                           if op.get("module_path") == path
                           or op.get("module_path", "").startswith(path + ".")]
            component_ms = component.get("estimated_healthy_total_ms")
            if (not descendants or not isinstance(component_ms, (int, float))
                    or isinstance(component_ms, bool) or not math.isfinite(float(component_ms))
                    or component_ms <= 0):
                continue
            framed_work_counts[family] += 1
            ranks_by_family[family].add(rank)
            work_operator_counts[family] += len(descendants)
            if len(site_examples[family]) < 3:
                site_examples[family].append({"rank": rank, "path": path,
                                              "module_class": component.get("module_class"),
                                              "framed_fwd_work_operator_sites": len(descendants),
                                              "example_operator": descendants[0].get("operator")})
            try:
                dose = FAMILY_HELPER.dose_from_catalog_site(component, baseline, 0.80,
                                                            reference_steps=3)
                site_dose_ratios[family].append(dose["slowdown_extra_ratio"])
            except ValueError:
                pass

    coverage = {family: {"candidate_sites": framed_work_counts[family],
                         "eligible_family_rows_before_work_filter": raw_counts[family],
                         "ranks_with_sites": sorted(ranks_by_family[family]),
                         "rank_count": len(ranks_by_family[family]),
                         "framed_fwd_work_operator_site_count": work_operator_counts[family],
                         "example_sites": site_examples[family],
                         "max_initial_ratio_at_target_fraction_0_80": max(site_dose_ratios[family], default=None)}
                for family in FAMILIES}
    if any(coverage[family]["candidate_sites"] == 0 for family in FAMILIES):
        raise ValueError(f"at least one semantic family has no nonzero framed FWD candidates: {coverage}")
    if any(coverage[family]["rank_count"] != 16 for family in FAMILIES):
        raise ValueError(f"each family must be rank-matchable across all 16 GPUs: {coverage}")
    return coverage, dict(unresolved_shapes)


def _component_pools(catalogs):
    pools = {}
    for rank, catalog in catalogs.items():
        valid_ops = [op for op in catalog.get("operators", [])
                     if _is_nonzero_framed_fwd_operator(op)[0]]
        rank_pools = {family: [] for family in FAMILIES}
        for component in catalog.get("components", []):
            family = FAMILY_HELPER.component_family(component)
            if family not in FAMILIES or _explicitly_excluded_component(component):
                continue
            path, slot = component.get("path"), component.get("slot_id")
            if (not component.get("eligible") or not isinstance(path, str) or not path
                    or slot != path):
                continue
            descendants = [op for op in valid_ops
                           if op.get("module_path") == path
                           or op.get("module_path", "").startswith(path + ".")]
            total_ms = component.get("estimated_healthy_total_ms")
            if (descendants and isinstance(total_ms, (int, float)) and not isinstance(total_ms, bool)
                    and math.isfinite(float(total_ms)) and total_ms > 0):
                rank_pools[family].append(component)
        pools[rank] = rank_pools
    return pools


def _catalog_refs_by_rank(layout):
    amendment = read_json(CATALOG_AMENDMENT)
    entries = {(row["layout"], int(row["rank"])): row for row in amendment.get("catalogs", [])}
    att = read_json(OLD_DESIGN)["scope_attestation"]
    refs = {}
    for rank in RANKS:
        row = entries[(layout, rank)]
        original_ref = att["catalogs"][layout][str(rank)]
        active_path = ROOT / f"results/data16_sessions/v2s_scope_guard_catalog_{layout}_01/scope_catalog_rank{rank}.json"
        refs[rank] = {
            "catalog_path": str(active_path.relative_to(ROOT)),
            "catalog_file_sha256": sha(active_path),
            "catalog_sha256": read_json(active_path).get("catalog_sha256"),
            "proof_path": original_ref["proof_path"],
            "proof_sha256": original_ref["proof_sha256"],
            "derived_from_catalog_path": row["old_path"],
            "catalog_amendment_sha256": sha(CATALOG_AMENDMENT),
        }
    return refs


def _make_cases(layout, rng, old_design, catalogs, refs, topology_ref, topology):
    pools = _component_pools(catalogs)
    active_ranks = {family: [rank for rank in RANKS if pools[rank][family]] for family in FAMILIES}
    if any(not active_ranks[family] for family in FAMILIES):
        raise ValueError(f"no eligible rank remains for a blind8 family in {layout}")
    rank_order = rng.sample(list(RANKS), k=len(FAMILIES))
    # The audited catalogs have candidates in all ranks for each stratum. Keep
    # one case per family on distinct ranks; randomize the family-to-rank map.
    for family, rank in zip(FAMILIES, rank_order):
        if rank not in active_ranks[family]:
            raise ValueError(f"rank matching changed after candidate audit: {layout} {family} rank {rank}")
    baseline = UPPER_PREPARE.base.old_whole_baseline(old_design, layout)
    cases = []
    for family, rank in zip(FAMILIES, rank_order):
        site = copy.deepcopy(rng.choice(pools[rank][family]))
        target_fraction = rng.uniform(0.40, 0.80)
        dose = FAMILY_HELPER.dose_from_catalog_site(site, baseline, target_fraction, reference_steps=3)
        case_id = f"blind8-{layout}-{family}-r{rank}-{rng.getrandbits(32):08x}"
        cases.append({
            "case_id": case_id, "layout": layout, "layer": "component_blind8",
            "category": family, "family": family, "rank": rank, "phase": "fwd",
            "population_size": len(pools[rank][family]),
            "family_population_total": sum(len(pools[r][family]) for r in RANKS),
            "population_basis": "all source-bound healthy catalog FWD component scopes with at least one fully timed framed descendant ATen site and all-positive tensor-shape signature; one family stratum per rank, no PROFILE filter",
            "site": site,
            "fault_target": {"kind": "component", "phase": "fwd", "slot_id": site["slot_id"]},
            "rank_catalog": refs[rank], "dose": dose,
            "training_seed": None, "topology": topology[rank],
            "topology_source": {"path": topology_ref["path"], "sha256": topology_ref["sha256"]},
        })
    if len(cases) != 4 or len({row["rank"] for row in cases}) != 4:
        raise AssertionError(f"blind8 layout rank/family assignment invalid: {layout}")
    if Counter(row["family"] for row in cases) != Counter({family: 1 for family in FAMILIES}):
        raise AssertionError(f"blind8 layout family quotas invalid: {layout}")
    return cases


def _make_plan(layout, mode, cases, design_sha, source_sha, design, peer_model):
    plan = UPPER_PREPARE.make_plan("l2", layout, mode, cases, design_sha,
                                   source_sha, design, peer_model)
    plan.update({
        "protocol": f"peer_priority_blind8_{mode}_v1",
        "session_id": f"v2s_peer_blind8_{mode}_{layout}_01",
        "role": f"component_blind8_{mode}",
        "layer": "component_blind8", "fault_kind": "component_compute",
        "fixed_denominator": 8,
        "component_operator_population_phase": "fwd",
        "component_family_quotas_per_layout": dict(QUOTAS),
        "scoped_reference_session": f"v2s_scope_guard_catalog_{layout}_01",
        "whole_compute_reference_session": f"v2s_scope_catalog_{layout}_02",
        "max_profiles": 0 if mode == "timeonly" else 16,
        "candidate_population_note": "one framed FWD component scope from each of the four declared semantic strata per layout; no whole/stage/operator-only cases; categories/sites/dose are held in the design and never passed to the localizer",
        "formal_execution_gate": "both layout timeonly healthy-stability gates must be stable; fixed targets and doses never change",
    })
    return plan


def make_timeonly():
    expected_files = ["design.json", "timeonly_legacy_plan.json", "timeonly_node_local_pp_plan.json",
                      "formal_legacy_plan.json", "formal_node_local_pp_plan.json",
                      "design_lock.json", "timeonly_ready.json", "plan_ready.json"]
    present = [name for name in expected_files if (HERE / name).exists()]
    if present:
        raise FileExistsError(f"preserve existing blind8 artifacts: {present}")
    candidate_audit = audit()
    freeze, peer_model, _ = _load_and_verify_freeze()
    source_sha = sha(SOURCE_FREEZE)
    old_design = read_json(OLD_DESIGN)
    att = old_design["scope_attestation"]
    attestation_summary = read_json(ROOT / att["path"])
    if attestation_summary.get("localizer_outputs_used") is not False:
        raise ValueError("blind8 candidates must not use localizer outputs")

    seed_hex = secrets.token_hex(16)
    rng = random.Random(int(seed_hex, 16))
    training_seed = rng.randrange(1, 2**31 - 1)
    design = {
        "protocol": "peer_priority_component_blind8_design_v1",
        "layer": "component_blind8", "integration_not_blind": False,
        "independent_randomized_acceptance": True, "prediction_before_truth": True,
        "design_seed_hex": seed_hex, "training_seed": training_seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_freeze": {"path": str(SOURCE_FREEZE.relative_to(ROOT)), "sha256": source_sha,
                          "core_model_sha256": freeze["model_sha256"],
                          "peer_model_sha256": peer_model.get("sha256")},
        "candidate_audit": {"path": str((HERE / "candidate_audit.json").relative_to(ROOT)),
                             "sha256": sha(HERE / "candidate_audit.json") if (HERE / "candidate_audit.json").exists() else canonical_hash(candidate_audit)},
        "catalog_source_amendment": {"path": str(CATALOG_AMENDMENT.relative_to(ROOT)),
                                     "sha256": sha(CATALOG_AMENDMENT)},
        "scope_attestation": {"path": att["path"], "sha256": att["sha256"],
                              "localizer_outputs_used": False},
        "execution_prerequisite": {
            "full_layer_revalidation": ["peer_priority_v1/l1", "peer_priority_v1/l2"],
            "must_pass_before_any_blind8_gpu": True,
            "required_gate": "each prior fixed-32 stage completed with 32/32 strict success and zero healthy wrong-GPU localizations",
            "prior_scores_are_not_design_inputs": True,
        },
        "fixed_denominator": 8, "cases_per_layout": 4,
        "family_quotas_per_layout": dict(QUOTAS),
        "component_operator_population_phase": "fwd_only",
        "dose_policy": "one independent U(0.40,0.80) target fraction per site; catalog healthy total divided by 3; one 87-step timeonly pass; no retuning, replacement, or localization-informed changes",
        "schedule": {"timeonly_steps_per_layout": TIMEONLY_STEPS,
                     "formal_steps_per_layout": FORMAL_STEPS,
                     "timeonly": "12 warmup + 30 healthy stability + 4*(5 healthy + 5 fault) + 5 tail",
                     "formal": "42 healthy + 4*(32 fault + 30 healthy) + 15 tail"},
        "scope_limitations": ["component sites are FWD-only", "family labels are sampling strata, not claims of whole semantic blocks",
                              "nonzero work requires every tensor in at least one recorded framed descendant signature to have positive numel"],
        "layouts": {},
    }

    catalog_data = {}
    topo_data = {}
    for layout in LAYOUTS:
        catalogs, refs = _catalogs_for_layout(layout, freeze, _verify_catalog_lineage(freeze))
        # Re-audit after no truth-informed inputs; guarantees every sampled pool
        # obeys the same corrected nonzero-work predicate as candidate_audit.
        family_coverage, _ = _audit_components(
            catalogs, UPPER_PREPARE.base.old_whole_baseline(old_design, layout))
        if any(family_coverage[f]["rank_count"] != 16 for f in FAMILIES):
            raise ValueError(f"family rank coverage changed after audit: {layout}")
        topo_ref, topology = UPPER_PREPARE.base.topology_by_rank(old_design, layout)
        cases = _make_cases(layout, rng, old_design, catalogs, refs, topo_ref, topology)
        for case in cases:
            case["training_seed"] = training_seed
        design["layouts"][layout] = cases
        catalog_data[layout] = (catalogs, refs)
        topo_data[layout] = (topo_ref, topology)

    if sum(len(rows) for rows in design["layouts"].values()) != 8:
        raise AssertionError("blind8 fixed denominator must remain 8")

    # Build every plan in memory first; the formal schedule is frozen now but
    # cannot be made runnable until finalize() accepts the 30-step health gate.
    design_path = HERE / "design.json"
    design_payload = json.dumps(design, indent=2, sort_keys=True, allow_nan=False) + "\n"
    design_sha = hashlib.sha256(design_payload.encode()).hexdigest()
    plan_payloads = {}
    for layout in LAYOUTS:
        for mode in ("timeonly", "formal"):
            plan = _make_plan(layout, mode, design["layouts"][layout], design_sha,
                              source_sha, design, peer_model)
            intervals = plan.get("intervals", [])
            faults = [row for row in intervals if row.get("segment") == "fault"]
            expected_steps = TIMEONLY_STEPS if mode == "timeonly" else FORMAL_STEPS
            expected_fault = 5 if mode == "timeonly" else 32
            if (plan.get("steps") != expected_steps or len(faults) != 4
                    or any(row["end"] - row["start"] + 1 != expected_fault for row in faults)):
                raise AssertionError(f"blind8 {mode} plan failed fixed schedule audit: {layout}")
            plan_payloads[f"{mode}_{layout}_plan.json"] = json.dumps(
                plan, indent=2, sort_keys=True, allow_nan=False) + "\n"

    lock_names = ["design.json", "timeonly_legacy_plan.json", "timeonly_node_local_pp_plan.json",
                  "formal_legacy_plan.json", "formal_node_local_pp_plan.json"]
    all_payloads = {"design.json": design_payload, **plan_payloads}
    design_lock = {"status": "design_and_both_schedules_locked_no_gpu_ready",
                   "stage": "component_blind8", "fixed_denominator": 8,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_freeze_sha256": source_sha, "design_sha256": design_sha,
                   "timeonly_steps_per_layout": TIMEONLY_STEPS,
                   "formal_steps_per_layout": FORMAL_STEPS,
                   "gpu_authorized": False,
                   "artifacts": {name: hashlib.sha256(all_payloads[name].encode()).hexdigest()
                                 for name in lock_names}}
    all_payloads["design_lock.json"] = json.dumps(
        design_lock, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for name, payload in all_payloads.items():
        path = HERE / name
        if path.exists():
            raise FileExistsError(f"preserve existing blind8 artifact: {path}")
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(payload)
    # Publish plans/design first and the non-executable lock marker last.
    for name in all_payloads:
        (HERE / f"{name}.tmp").replace(HERE / name)
    return design_lock


def require_prior_full_layer_pass():
    checks = {}
    for stage in ("l1", "l2"):
        path = PARENT / stage / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"blind8 GPU remains locked until peer-priority full {stage.upper()} score exists")
        summary = read_json(path)
        denominator = 32
        healthy_wrong = summary.get('healthy_localized_false_alarms')
        if (type(healthy_wrong) is not int or healthy_wrong != 0
                or summary.get("fixed_denominator") != denominator
                or summary.get("strict_success") != denominator
                or summary.get("gate_passed") is not True
                ):
            raise RuntimeError(f"peer-priority full {stage.upper()} must pass 32/32 with zero healthy wrong-GPU cases")
        checks[stage] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                         "strict_success": denominator, "fixed_denominator": denominator,
                         "healthy_localized_false_alarms": healthy_wrong}
    return checks


def authorize_timeonly():
    lock_path = HERE / "design_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("run make-timeonly to lock the blind8 seed/design first")
    lock = read_json(lock_path)
    if lock.get("gpu_authorized") is not False or lock.get("fixed_denominator") != 8:
        raise ValueError("invalid blind8 design lock")
    if sha(SOURCE_FREEZE) != lock.get("source_freeze_sha256"):
        raise ValueError("source freeze changed after blind8 design draw")
    for name, digest in lock.get("artifacts", {}).items():
        if not (HERE / name).is_file() or sha(HERE / name) != digest:
            raise ValueError(f"blind8 locked design artifact changed: {name}")
    prior_gate = require_prior_full_layer_pass()
    ready_path = HERE / "timeonly_ready.json"
    if ready_path.exists():
        raise FileExistsError(f"preserve existing blind8 time-only ready marker: {ready_path}")
    names = ["source_freeze.json", "design_lock.json", *lock["artifacts"].keys()]
    ready = {"status": "timeonly_ready_after_prior_layer_gates",
             "stage": "component_blind8", "fixed_denominator": 8,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "source_freeze_sha256": sha(SOURCE_FREEZE),
             "design_sha256": sha(HERE / "design.json"),
             "prior_full_layer_gate": prior_gate,
             "artifacts": {name: sha(SOURCE_FREEZE if name == "source_freeze.json" else HERE / name)
                           for name in names}}
    temp = HERE / "timeonly_ready.json.tmp"
    temp.write_text(json.dumps(ready, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(HERE / "timeonly_ready.json")
    return ready


def finalize():
    ready_path = HERE / "timeonly_ready.json"
    gate_path = HERE / "timeonly_gate.json"
    if not ready_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError("blind8 timeonly_ready.json and timeonly_gate.json are required")
    ready, gate = read_json(ready_path), read_json(gate_path)
    if (ready.get("fixed_denominator") != 8 or gate.get("fixed_denominator") != 8
            or gate.get("stable_for_formal") is not True
            or gate.get("source_freeze_sha256") != ready.get("source_freeze_sha256")
            or gate.get("design_sha256") != ready.get("design_sha256")
            or gate.get("timeonly_ready_sha256") != sha(ready_path)):
        raise RuntimeError("blind8 formal plans remain locked until both health stability gates pass")
    prior_gate = require_prior_full_layer_pass()
    if ready.get("source_freeze_sha256") != sha(SOURCE_FREEZE):
        raise ValueError("source freeze changed after blind8 seed/design draw")
    for name, digest in ready.get("artifacts", {}).items():
        path = SOURCE_FREEZE if name == "source_freeze.json" else HERE / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"blind8 timeonly/design artifact changed: {name}")
    final_ready_path = HERE / "plan_ready.json"
    if final_ready_path.exists():
        raise FileExistsError(f"preserve existing blind8 formal ready marker: {final_ready_path}")
    design = read_json(HERE / "design.json")
    for layout in LAYOUTS:
        plan = read_json(HERE / f"formal_{layout}_plan.json")
        if (plan.get("steps") != FORMAL_STEPS or plan.get("fixed_denominator") != 8
                or plan.get("layout") != layout):
            raise ValueError(f"frozen blind8 formal plan changed: {layout}")
    names = ["source_freeze.json", "design.json", "timeonly_ready.json", "timeonly_gate.json",
             "formal_legacy_plan.json", "formal_node_local_pp_plan.json"]
    final_ready = {"status": "ready_for_single_gpu_owner_after_layer_gates",
                   "stage": "component_blind8", "fixed_denominator": 8,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_freeze_sha256": sha(SOURCE_FREEZE),
                   "design_sha256": sha(HERE / "design.json"),
                   "timeonly_gate_sha256": sha(gate_path),
                   "prior_full_layer_gate_required": True,
                   "prior_full_layer_gate": prior_gate,
                   "artifacts": {name: sha(SOURCE_FREEZE if name == "source_freeze.json" else HERE / name)
                                 for name in names}}
    temp = HERE / "plan_ready.json.tmp"
    temp.write_text(json.dumps(final_ready, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(HERE / "plan_ready.json")
    return final_ready


def _schedule_audit():
    cases = [{"case_id": f"audit-{family}", "rank": index,
              "fault_target": {"kind": "component", "phase": "fwd", "slot_id": f"site-{family}"},
              "dose": {"slowdown_extra_ratio": 1.0}}
             for index, family in enumerate(FAMILIES)]
    time_intervals, time_steps = UPPER_PREPARE.intervals_for("l2", cases, "timeonly")
    formal_intervals, formal_steps = UPPER_PREPARE.intervals_for("l2", cases, "formal")
    for mode, intervals, steps, expected, fault_steps in (
            ("timeonly", time_intervals, time_steps, TIMEONLY_STEPS, 5),
            ("formal", formal_intervals, formal_steps, FORMAL_STEPS, 32)):
        faults = [row for row in intervals if row.get("segment") == "fault"]
        if steps != expected or intervals[-1].get("end") != expected or len(faults) != 4:
            raise AssertionError(f"unexpected {mode} blind8 schedule")
        if any(row["end"] - row["start"] + 1 != fault_steps for row in faults):
            raise AssertionError(f"unexpected {mode} fault duration")
        if any(row.get("fault_mode") != "whole_compute" or row.get("fault_scope", {}).get("kind") != "component"
               for row in faults):
            raise AssertionError(f"unexpected {mode} fault target kind")
    return {"timeonly_steps_per_layout": TIMEONLY_STEPS,
            "formal_steps_per_layout": FORMAL_STEPS,
            "case_count_per_layout": 4,
            "timeonly": "12 warmup + 30 healthy + 4*(5 healthy + 5 fault) + 5 tail",
            "formal": "42 healthy + 4*(32 fault + 30 healthy) + 15 tail"}


def audit():
    predicate_examples = _assert_nonzero_signature_examples()
    freeze, peer_model, peer_binding = _load_and_verify_freeze()
    amendment = _verify_catalog_lineage(freeze)
    attest_design = read_json(OLD_DESIGN)
    att = attest_design.get("scope_attestation", {})
    att_path = ROOT / att.get("path", "")
    if (not att_path.is_file() or sha(att_path) != att.get("sha256")
            or att.get("status") != "passed" or att.get("localizer_outputs_used") is not False):
        raise ValueError("independent GPU-validated catalog attestation is unavailable or changed")

    report = {
        "protocol": "peer_priority_v1_component_blind8_candidate_audit",
        "status": "candidate_population_and_schedule_audited_no_design_drawn",
        "writes_random_design": False,
        "draws_seed": False,
        "source_freeze": {"path": str(SOURCE_FREEZE.relative_to(ROOT)), "sha256": sha(SOURCE_FREEZE),
                          "remote_source_check": str((PARENT / "remote_source_check.json").relative_to(ROOT)),
                          "core_model_sha256": freeze.get("model_sha256"),
                          "scoped_peer_model_sha256": peer_model.get("sha256"),
                          "scoped_peer_binding_sha256": sha(AUX_BINDING_PATH)},
        "catalog_lineage": {"amendment_path": str(CATALOG_AMENDMENT.relative_to(ROOT)),
                            "amendment_sha256": sha(CATALOG_AMENDMENT),
                            "active_catalog_session_suffix": "v2s_scope_guard_catalog_{layout}_01",
                            "population_traces_times_and_signatures_preserved": True,
                            "new_gpu_validation_claimed": False},
        "population_contract": {
            "sample_unit": "one eligible FWD component scope per semantic family per layout",
            "families_per_layout": list(FAMILIES), "quota_per_family_per_layout": QUOTAS,
            "fixed_denominator_two_layouts": 8, "cases_per_layout": 4,
            "requires_fwd_framed_descendant_aten_site_with_timed_positive_tensor_signature": True,
            "excludes_whole_model_stage_model_and_structural_containers": True,
            "component_operator_population_phase": "fwd_only",
            "family_interpretation_limit": "catalog component scopes can be parent modules or descendants; labels are sampling strata, not claims of complete semantic blocks",
            "selection_independent_of_PROFILE_and_localizer_truth": True,
            "nonzero_signature_boundary_examples": predicate_examples,
        },
        "schedule": _schedule_audit(),
        "dose_policy_predeclared": "per-site healthy catalog estimate uses 3-step total / 3; target fraction U(0.40,0.80); one time-only confirmation; no retune/replacement or localization-informed selection",
        "planned_gate": {"strict_success_required": 8, "fixed_denominator": 8,
                         "scoring_basis": "fault-window trigger, correct target, complete 8-step evidence/profile, valid scope/FMA; retain reject, miss, weak and invalid cases",
                         "report_separately": ["core vs peer source", "healthy triggers/wrong GPU", "peer support coverage", "actual continuation-child count", "time-only effect"]},
        "layouts": {},
    }
    for layout in LAYOUTS:
        catalogs, catalog_refs = _catalogs_for_layout(layout, freeze, amendment)
        baseline = UPPER_PREPARE.base.old_whole_baseline(attest_design, layout)
        family_coverage, filtered_operator_rows = _audit_components(catalogs, baseline)
        report["layouts"][layout] = {
            "rank_count": len(catalogs), "reference_healthy_wall_ms": baseline,
            "component_operator_phases": ["fwd"],
            "candidate_family_coverage": family_coverage,
            "catalog_references": catalog_refs,
            "noncandidate_operator_rows_excluded_for_signature_or_timing": filtered_operator_rows,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="audit catalogs/schedule only; no seed/design")
    parser.add_argument("--write-audit", action="store_true", help="save deterministic candidate_audit.json")
    parser.add_argument("command", nargs="?", choices=("make-timeonly", "authorize-timeonly", "finalize"))
    args = parser.parse_args()
    if args.command:
        if args.check or args.write_audit:
            parser.error("commands cannot be combined with --check/--write-audit")
        if args.command == "make-timeonly":
            result = make_timeonly()
        elif args.command == "authorize-timeonly":
            result = authorize_timeonly()
        else:
            result = finalize()
    else:
        if not args.check and not args.write_audit:
            parser.error("use --check, --write-audit, or a preparation command")
        result = audit()
    if args.write_audit:
        HERE.mkdir(parents=True, exist_ok=True)
        output = HERE / "candidate_audit.json"
        temp = output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temp.replace(output)
        result["audit_artifact"] = str(output.relative_to(ROOT))
        result["audit_artifact_sha256"] = sha(output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
