#!/usr/bin/env python3
"""Independently score the fixed reg8 sparse3 + startup-GC candidate.

All public controller events are checked and sealed before truth/evidence files
are opened.  Scoring logic is reused from the immutable reverted-baseline
scorer; this file only verifies the candidate-specific source and session
bindings and applies the predeclared regression gate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_reverted_baseline"
STAGED_PRODUCTION = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_sparse3_gc_lifetime_v2/production"
LAYOUTS = ("legacy", "node_local_pp")
ALLOWED = {"detection", "profile_schedule", "localization"}
RANKS = set(range(16))
SPARSE_OFFSETS = (1, 3, 7)


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_scorer():
    path = OLD / "score.py"
    spec = importlib.util.spec_from_file_location("reg8_fixed_score_reuse", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed scoring implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_persisted_plan(plan):
    sys.path.insert(0, str(STAGED_PRODUCTION))
    from flash_v1.startup_gc import configure_plan
    return configure_plan(plan)


def install_sparse_auxiliary_check(scorer):
    """Use the frozen sparse3 infer path; never pretend four sampled steps are eight."""
    sys.path.insert(0, str(STAGED_PRODUCTION))
    from flash_v1.stage_forward_localize import validate_row
    from flash_v1.stage_forward_peer import verify_topology
    from flash_v1.stage_forward_sampling import contract as sampling_contract
    from flash_v1.stage_forward_sparse_peer import infer_bound

    def check(schedule, all_auxiliary, session, layout, model, binding):
        if schedule is None or int(schedule.get("end", -1)) - int(schedule.get("start", -1)) + 1 != 8:
            return {"complete": False, "rows": 0, "expected_rows": 48,
                    "reason": "missing_or_non_eight_step_schedule"}
        start, end, epoch = int(schedule["start"]), int(schedule["end"]), int(schedule["epoch"])
        wanted_steps = {start + offset - 1 for offset in SPARSE_OFFSETS}
        rows = [row for row in all_auxiliary if row.get("epoch") == epoch
                and start <= row.get("iteration", -1) <= end]
        required = {(iteration, rank) for iteration in wanted_steps for rank in RANKS}
        identities = [(row.get("iteration"), row.get("rank")) for row in rows]
        errors = []
        expected_contract = sampling_contract(epoch, start)
        for row in rows:
            try:
                if row.get("record_type") != "stage_forward_aux":
                    raise ValueError("invalid_auxiliary_record_type")
                if row.get("sampling_contract") != expected_contract:
                    raise ValueError("sparse_sampling_contract_mismatch")
                validate_row(row)
            except Exception as exc:
                errors.append(str(exc))
        topology_files = sorted(session.glob("host*/topology.json"))
        topology_ok = False
        topology_sha = None
        try:
            if not topology_files:
                raise ValueError("missing_topology")
            topologies = [scorer.read(path) for path in topology_files]
            canonical = scorer.upper.canonical_sha256
            topology_sha = canonical(topologies[0])
            if any(canonical(value) != topology_sha for value in topologies[1:]):
                raise ValueError("host_topologies_differ")
            verify_topology(model, topologies[0], layout)
            topology_ok = True
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"topology:{exc}")
        complete = (len(rows) == 48 and len(set(identities)) == 48
                    and set(identities) == required and not errors and topology_ok)
        peer = None
        if complete:
            peer = infer_bound(rows, model, RANKS, binding)
        pair_support = (peer or {}).get("pair_support", {})
        support = {str(pair): {
            "matched_iterations": int(detail.get("matched_iterations", 0)),
            "unknown_iterations": int(detail.get("unknown_iterations", 0)),
            "comparable": detail.get("comparable") is True,
        } for pair, detail in pair_support.items()}
        known_pairs = sum(detail["matched_iterations"] >= 3 for detail in support.values())
        return {
            "schedule": {"epoch": epoch, "start": start, "end": end,
                         "sample_offsets": list(SPARSE_OFFSETS)},
            "rows": len(rows), "expected_rows": 48,
            "unique_rank_iteration_pairs": len(set(identities)),
            "sampled_steps": sorted(wanted_steps),
            "unsampled_steps_are_healthy": False,
            "topology_valid": topology_ok, "topology_sha256": topology_sha,
            "validation_errors": errors[:8], "complete": complete,
            "status": (peer or {}).get("status", "not_inferred"),
            "rank": (peer or {}).get("rank"), "reason": (peer or {}).get("reason"),
            "support_by_pair": support, "known_pairs_support3": known_pairs,
            "result": peer,
        }

    scorer.upper.auxiliary_check = check


def install_sparse_layout_scorer(scorer):
    """Derive the audited reg8 case scorer with only its explicit sparse row count changed."""
    source = inspect.getsource(scorer.score_layout)
    old = 'aux.get("rows") == 128 and aux.get("expected_rows") == 128'
    new = 'aux.get("rows") == 48 and aux.get("expected_rows") == 48'
    if source.count(old) != 1:
        raise ValueError("parent scorer sparse3 adaptation point changed")
    namespace = dict(scorer.__dict__)
    exec(compile(source.replace(old, new), str(OLD / "score.py") + "[sparse3-row-contract]", "exec"), namespace)
    scorer.score_layout = namespace["score_layout"]


def verify_and_freeze_public():
    ready_path = HERE / "ready.json"
    freeze_path = HERE / "source_freeze.json"
    design_path = HERE / "design.json"
    complete_path = HERE / "execution_complete.json"
    remote_path = HERE / "remote_candidate_source_check.json"
    public_index_path = HERE / "public_predictions_before_truth.json"
    for path in (ready_path, freeze_path, design_path, complete_path, remote_path, public_index_path):
        if not path.is_file():
            raise FileNotFoundError(f"candidate scoring requires {path.name}")

    ready, freeze, design, complete = map(read, (ready_path, freeze_path, design_path, complete_path))
    remote, public_index = read(remote_path), read(public_index_path)
    source_sha = sha(freeze_path)
    if (ready.get("status") != "ready_for_new_fixed_case_sparse3_gcdiag_regression"
            or ready.get("fixed_denominator") != 8
            or ready.get("source_freeze_sha256") != source_sha
            or design.get("source_freeze", {}).get("sha256") != source_sha
            or complete.get("status") != "complete"
            or complete.get("fixed_denominator") != 8
            or complete.get("steps_per_layout") != 305
            or complete.get("source_freeze_sha256") != source_sha
            or public_index.get("prediction_before_truth") is not True
            or public_index.get("fixed_denominator") != 8
            or public_index.get("source_freeze_sha256") != source_sha):
        raise ValueError("candidate ready/completion/source identity mismatch")

    for name, expected in ready.get("artifacts", {}).items():
        path = freeze_path if name == "source_freeze.json" else HERE / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"candidate ready artifact hash mismatch: {name}")

    baseline_design = read(OLD / "design.json")
    if (design.get("training_seed") != baseline_design.get("training_seed")
            or design.get("layouts") != baseline_design.get("layouts")
            or design.get("fixed_denominator") != 8
            or design.get("integration_not_blind") is not True):
        raise ValueError("candidate changed the predeclared fixed cases or seed")

    staged_tree = HERE / "staged_tree.json"
    staged_tar = staged_tree.with_name("staged_flash_v1.tar")
    expected_staged = read(staged_tree).get("tar_sha256")
    if sha(staged_tar) != expected_staged:
        raise ValueError("candidate staged production bundle changed")
    for host in ("host1", "host2"):
        check = remote.get(host, {})
        if check.get("status") != "candidate_overlay_ready":
            raise ValueError(f"candidate staged source not verified on {host}")
        actual = check.get("source_sha256", {})
        expected = {**freeze.get("files", {}), **freeze.get("binary_files", {})}
        if actual != expected:
            raise ValueError(f"candidate staged source hash mismatch on {host}")

    prior_candidate = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_sparse4_gc_lifetime_v2"
    baseline_plans = {layout: read(prior_candidate / f"formal_{layout}_plan.json") for layout in LAYOUTS}
    sessions = {}
    prediction_layouts = {}
    for layout in LAYOUTS:
        plan_path = HERE / f"formal_{layout}_plan.json"
        plan = read(plan_path)
        parent = baseline_plans[layout]
        changed = {k for k in set(plan) | set(parent) if plan.get(k) != parent.get(k)}
        if changed != {"session_id", "freeze_manifest_sha256", "freeze_manifest_path",
                       "stage_forward_aux_sampling", "gc_callbacks_diagnostic", "monitor_cost_timing"}:
            raise ValueError(f"candidate {layout}: plan changed beyond source/session binding: {sorted(changed)}")
        if (plan.get("steps") != 305 or plan.get("seed") != design.get("training_seed")
                or plan.get("freeze_manifest_sha256") != source_sha
                or plan.get("freeze_manifest_path") != str(freeze_path.relative_to(ROOT))
                or plan.get("layout") != layout or plan.get("parallel_layout") != layout):
            raise ValueError(f"candidate {layout}: frozen plan identity mismatch")
        completion = complete.get("sessions", {}).get(layout, {})
        if completion.get("session_id") != plan.get("session_id"):
            raise ValueError(f"candidate {layout}: execution completion points at another session")
        session = ROOT / "results/data16_sessions" / plan["session_id"]
        if completion.get("session") != str(session.relative_to(ROOT)):
            raise ValueError(f"candidate {layout}: execution completion session path mismatch")
        for host, manifest_name, node_rank in (
            ("host1", "node1_manifest.json", 1), ("host2", "node0_manifest.json", 0)):
            host_dir = session / host
            if read(host_dir / "plan.json") != configure_persisted_plan(plan):
                raise ValueError(f"candidate {layout}/{host}: persisted plan differs")
            manifest = read(host_dir / manifest_name)
            config = manifest.get("configuration", {})
            if (manifest.get("status") != "completed"
                    or config.get("startup_gc_fixed_model") is not True
                    or config.get("startup_gc_policy", {}).get("schema") != "startup_process_lifetime_heap_freeze_v2"
                    or config.get("stage_forward_aux_sampling") != plan.get("stage_forward_aux_sampling")
                    or config.get("gc_callbacks_diagnostic") is not True):
                raise ValueError(f"candidate {layout}/{host}: incomplete training or missing startup-GC policy")
            gc_paths = sorted((host_dir).glob("gc_iteration_diagnostic_rank*.json"))
            if len(gc_paths) != 8:
                raise ValueError(f"candidate {layout}/{host}: expected eight GC diagnostic rank files")
            for gc_path in gc_paths:
                record = read(gc_path)
                if (record.get("schema") != "whole_iteration_gc_diagnostic_v1"
                        or record.get("status") != "complete"
                        or record.get("no_GPU_sync_or_GC_policy_change") is not True):
                    raise ValueError(f"candidate {layout}/{host}: invalid GC diagnostic {gc_path.name}")
            cost_paths = sorted(host_dir.glob("monitor_cost_rank*.json"))
            if len(cost_paths) != 8 or any(len(read(path)) != 305 for path in cost_paths):
                raise ValueError(f"candidate {layout}/{host}: incomplete monitor-cost traces")
        train_path = session / "host2/train.jsonl"
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
        if len(train_rows) != 305 or int(train_rows[-1].get("iteration", -1)) != 305:
            raise ValueError(f"candidate {layout}: merged training stream is incomplete")

        events_path = session / "host2/events.jsonl"
        all_events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        public_events = [row for row in all_events if row.get("event") in ALLOWED]
        pred_path = HERE / f"{layout}_predictions_without_truth.json"
        pred = read(pred_path)
        expected_pred = {
            "protocol": "peer_priority_reg8_sparse3_gcdiag_public_snapshot_v1",
            "layout": layout,
            "session": session.name,
            "prediction_before_truth": True,
            "source_freeze_sha256": source_sha,
            "plan_sha256": sha(plan_path),
            "public_events_sha256": sha(events_path),
            "events": public_events,
        }
        if pred != expected_pred or sha(pred_path) != completion.get("prediction_sha256"):
            raise ValueError(f"candidate {layout}: public prediction snapshot differs from events or completion")
        index = public_index.get("layouts", {}).get(layout, {})
        if (index.get("session_id") != session.name
                or index.get("plan_sha256") != sha(plan_path)
                or index.get("public_events_sha256") != sha(events_path)
                or index.get("prediction_sha256") != sha(pred_path)):
            raise ValueError(f"candidate {layout}: public prediction index mismatch")

        # Construct exactly the input contract expected by the already-audited scorer.
        sessions[layout] = {
            "session": session,
            "plan": plan,
            "design": design,
            "events": public_events,
            "prediction_sha256": sha(pred_path),
            "plan_sha256": sha(plan_path),
        }
        prediction_layouts[layout] = {
            "session": session.name,
            "events_sha256": sha(events_path),
            "prediction_sha256": sha(pred_path),
            "plan_sha256": sha(plan_path),
        }

    sealed = {
        "protocol": "peer_priority_reg8_sparse3_gcdiag_prediction_freeze_v1",
        "prediction_before_truth": True,
        "fixed_denominator": 8,
        "source_freeze_sha256": source_sha,
        "execution_complete_sha256": sha(complete_path),
        "layouts": prediction_layouts,
    }
    sealed_path = HERE / "independent_prediction_freeze.json"
    if sealed_path.exists() and read(sealed_path) != sealed:
        raise FileExistsError("preserve existing independent prediction freeze")
    if not sealed_path.exists():
        atomic_json(sealed_path, sealed)
    return {
        "source_sha256": source_sha,
        "freeze": freeze,
        "design": design,
        "sessions": sessions,
        "execution_complete_sha256": sha(complete_path),
        "prediction_freeze_path": sealed_path,
        "prediction_freeze_sha256": sha(sealed_path),
    }


def main():
    scorer = load_scorer()
    install_sparse_auxiliary_check(scorer)
    install_sparse_layout_scorer(scorer)
    bundle = verify_and_freeze_public()
    # Truth and evidence are opened only after all public predictions are sealed.
    layouts = {
        layout: scorer.score_layout(layout, bundle["sessions"][layout], bundle)
        for layout in LAYOUTS
    }
    summary = scorer.summarize(layouts)
    baseline = read(OLD / "summary.json")
    baseline_cases = {(row["layout"], row["case_id"]): row for row in baseline["cases"]}
    candidate_cases = {(row["layout"], row["case_id"]): row for row in summary["cases"]}
    prior_successes = sorted(key for key, row in baseline_cases.items() if row.get("end_to_end_success") is True)
    prior_successes_preserved = all(candidate_cases.get(key, {}).get("end_to_end_success") is True for key in prior_successes)
    critical_families = {"whole_compute", "backward_stage", "nonfirst_normalization_component"}
    critical_cases = [row for row in summary["cases"] if row.get("family") in critical_families]
    critical_success = sum(row.get("end_to_end_success") is True for row in critical_cases)
    health_wrong = summary["healthy_wrong_gpu_localizations"]
    summary.update({
        "candidate_protocol": "sparse3_gcdiag_fixed_reg8_candidate_score_v1",
        "candidate_source_freeze_sha256": bundle["source_sha256"],
        "candidate_execution_complete_sha256": bundle["execution_complete_sha256"],
        "independent_prediction_freeze_path": str(bundle["prediction_freeze_path"].relative_to(ROOT)),
        "independent_prediction_freeze_sha256": bundle["prediction_freeze_sha256"],
        "baseline_summary_sha256": sha(OLD / "summary.json"),
        "baseline_successful_cases": [
            {"layout": layout, "case_id": case_id} for layout, case_id in prior_successes
        ],
        "baseline_successes_preserved": prior_successes_preserved,
        "critical_whole_backward_component_cases": {
            "fixed_denominator": 6,
            "strict_success": critical_success,
            "case_ids": [row["case_id"] for row in critical_cases],
        },
        "candidate_gate_passed": bool(
            summary["strict_success"] >= len(prior_successes)
            and prior_successes_preserved
            and critical_success == 6
            and health_wrong == 0
        ),
        "candidate_gate_rule": "fixed denominator 8; all seven prior baseline successes preserved; all six whole/BWD/FWD-component cases strict-success; healthy wrong-GPU localizations zero; no weak, invalid, rejected or missed case removed",
    })
    atomic_json(HERE / "candidate_cases.json", {
        "protocol": summary["candidate_protocol"],
        "fixed_denominator": 8,
        "prediction_before_truth": True,
        "layouts": layouts,
    })
    atomic_json(HERE / "candidate_summary.json", summary)
    print(json.dumps({
        "summary": str((HERE / "candidate_summary.json").relative_to(ROOT)),
        "strict_success": summary["strict_success"],
        "fixed_denominator": 8,
        "baseline_successes_preserved": prior_successes_preserved,
        "critical_six_success": critical_success,
        "healthy_wrong_gpu_localizations": health_wrong,
        "candidate_gate_passed": summary["candidate_gate_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
