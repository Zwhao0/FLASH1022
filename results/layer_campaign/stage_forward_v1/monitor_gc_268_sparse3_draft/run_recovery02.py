"""One root-authorized MONITOR268 recovery with explicit fixed-model GC opt-in."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
LAYOUTS = ("legacy", "node_local_pp")
PUBLIC_EVENTS = {"detection", "profile_schedule", "localization", "quota_exhausted"}


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def preflight(layout: str):
    amendment = read(HERE / "recovery02_amendment.json")
    ready = read(HERE / "recovery02_ready.json")
    contract = read(HERE / "contract.json")
    freeze = read(HERE / "freeze.json")
    if (ready.get("amendment_sha256") != sha(HERE / "recovery02_amendment.json")
            or ready.get("contract_sha256") != sha(HERE / "contract.json")
            or ready.get("freeze_sha256") != sha(HERE / "freeze.json")
            or ready.get("runner_sha256") != sha(HERE / "run_recovery02.py")
            or amendment.get("original_infrastructure_failure_sha256") != sha(HERE / "execution_failure_legacy.json")
            or amendment.get("recovery01_failure_sha256") != sha(HERE / "recovery_failure_legacy.json")
            or amendment.get("original_attempt_has_measurement") is not False
            or amendment.get("recovery01_has_measurement") is not False
            or amendment.get("fixed_measurement_count") != 2):
        raise ValueError("recovery amendment/ready contract mismatch")
    if not (HERE / "execution_failure_legacy.json").is_file():
        raise FileNotFoundError("preserve the original infrastructure failure")
    original_entry = contract["plans"][layout]
    original = read(ROOT / original_entry["path"])
    recovery_entry = amendment["recovery_plans"][layout]
    recovery_path = HERE / recovery_entry["path"]
    recovery = read(recovery_path)
    if sha(ROOT / original_entry["path"]) != original_entry["sha256"]:
        raise ValueError(f"{layout}: original frozen plan hash changed")
    if (sha(recovery_path) != ready["recovery_plans"][layout]
            or recovery.get("session_id") != amendment["mappings"][layout]["recovery02_session_id"]
            or recovery_entry.get("sha256") != sha(recovery_path)):
        raise ValueError(f"{layout}: recovery plan/ready binding mismatch")
    # The explicit boolean and sparse sampling contract must match the actual
    # production configurator; no other plan fields may move except session_id.
    from flash_v1.startup_gc import configure_plan, validate_plan, POLICY
    from flash_v1.stage_forward_sampling import CONFIG
    configured = configure_plan(original)
    expected = dict(configured, session_id=recovery["session_id"])
    if recovery != expected:
        raise ValueError(f"{layout}: recovery differs from configure_plan(original) beyond fresh session ID")
    if (recovery.get("startup_gc_fixed_model") is not True
            or recovery.get("startup_gc_policy") != POLICY
            or recovery.get("stage_forward_aux_sampling") != CONFIG
            or validate_plan(recovery) is not True):
        raise ValueError(f"{layout}: production startup GC/aux plan validation failed")
    # Mirror the exact pre-launch checks performed by node_cost and its delegated
    # node_collection_cost entry, without invoking subprocesses or training.
    if (recovery.get("monitor_sameprocess_cost") is not True
            or recovery.get("steps") != 268 or recovery.get("max_profiles") != 3
            or recovery.get("cost_harness_files") != freeze["cost_harness_files"]
            or recovery.get("integration_node_module") !=
               "results.layer_campaign.stage_forward_v1.monitor_gc_268_sparse3_draft.node_cost"):
        raise ValueError(f"{layout}: actual node_cost entry contract mismatch")
    for source, expected_sha in recovery["cost_harness_files"].items():
        if sha(ROOT / source) != expected_sha:
            raise ValueError(f"{layout}: cost harness source mismatch: {source}")
    if (recovery.get("parallel_layout") != layout or recovery.get("seed") != contract["seed"]
            or recovery.get("controlled_profile_requests") != []
            or recovery.get("controlled_manual_profile_limit") != 0
            or recovery.get("profile_backend") != "semantic_only"
            or recovery.get("probe_enabled") is not False
            or recovery.get("evidence_enabled") is not True
            or recovery.get("stage_forward_aux_enabled") is not True
            or recovery.get("monitor_profile_transition") is not True
            or recovery.get("monitor_confirm_recent_three") is not True
            or recovery.get("gc_callbacks_diagnostic") is not False
            or recovery.get("monitor_cost_timing") is not False):
        raise ValueError(f"{layout}: cost/sampling/source metadata differs from locked contract")
    session_dir = ROOT / "results/data16_sessions" / recovery["session_id"]
    if session_dir.exists():
        raise FileExistsError(f"fresh recovery session already exists: {session_dir}")
    # Re-check the actual production source map immediately before launching.
    import importlib.util
    spec = importlib.util.spec_from_file_location("monitor268_original_runner", HERE / "run.py")
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    runner_module.check(staged=False)
    return recovery, freeze, amendment, contract


def execute(layout: str):
    recovery, freeze, amendment, contract = preflight(layout)
    start_path = HERE / f"recovery02_start_{layout}.json"
    if start_path.exists() or (HERE / f"recovery02_complete_{layout}.json").exists():
        raise FileExistsError(f"one recovery per layout only: {layout}")

    from results.layer_campaign.continuation_v2.gates import owned_runner as owner
    owner.D = HERE
    owner.freeze = freeze
    owner.check_sources()

    # Install only the frozen, isolated monitor backend. It never overwrites production files.
    backend = (HERE / "isolated_backend.tar").read_bytes()
    encoded = base64.b64encode(backend).decode("ascii")
    extraction = "import base64,io,tarfile;tarfile.open(fileobj=io.BytesIO(base64.b64decode(__import__('sys').stdin.read()))).extractall('.')"
    for host, container, sudo in owner.NODES:
        owner.rpc(host, owner.dock(host, container, sudo, ["python", "-c", extraction]), input=encoded)
    owner.check_sources()

    # The failed pre-recovery session is diagnostic only; it must contain no training/evidence.
    failed_sessions = [amendment["mappings"][layout]["original_session_id"],
                       amendment["mappings"][layout]["recovery01_session_id"]]
    check_old = "import json;from pathlib import Path;ss=" + repr(failed_sessions) + ";out={};\nfor s in ss:\n p=Path('results/data16_sessions')/s;out[s]={'bad':[x.name for x in p.glob('**/*') if x.is_file() and (x.name=='train.jsonl' or x.name.startswith('evidence_rank') or x.name.startswith('stage_forward_aux_rank') or x.name.startswith('whole_compute_truth_rank'))],'exists':p.exists()}\nprint(json.dumps(out))"
    old_check = {}
    for host, container, sudo in owner.NODES:
        old_check[host] = json.loads(owner.rpc(host, owner.dock(host, container, sudo, ["python", "-c", check_old])))
    if any(record["bad"] for value in old_check.values() for record in value.values()):
        raise ValueError("original stopped attempt contains training/evidence; do not treat it as no-measurement")

    save(start_path, {
        "status": "starting_root_authorized_infrastructure_recovery",
        "layout": layout,
        "recovery_session_id": recovery["session_id"],
        "recovery_amendment_sha256": sha(HERE / "recovery02_amendment.json"),
        "recovery_plan_sha256": sha(HERE / f"recovery02_{layout}_plan.json"),
        "contract_sha256": sha(HERE / "contract.json"),
        "source_freeze_sha256": sha(HERE / "freeze.json"),
        "prior_failed_sessions_empty_training_check": old_check,
        "no_automatic_retry": True,
    })
    try:
        owner.rendezvous_port = 30520 if layout == "legacy" else 30521
        session = owner.detached_run(recovery)
        # owned_runner returns the session root with each host's files nested
        # under host1/host2; rank-zero train timings live under host2.
        train = [json.loads(line) for line in (session / "host2/train.jsonl").read_text().splitlines() if line.strip()]
        if len(train) != 268 or int(train[-1].get("iteration", -1)) != 268:
            raise ValueError("recovered monitor train stream is incomplete")
        for host, manifest_name in (("host1", "node1_manifest.json"), ("host2", "node0_manifest.json")):
            manifest = read(session / host / manifest_name)
            actual_plan = read(session / host / "plan.json")
            if manifest.get("status") != "completed":
                raise ValueError(f"{host}: recovered monitor node did not complete")
            if (actual_plan.get("session_id") != recovery["session_id"]
                    or actual_plan.get("intervals") != recovery.get("intervals")
                    or actual_plan.get("seed") != recovery.get("seed")
                    or actual_plan.get("steps") != 268):
                raise ValueError(f"{host}: recovered plan semantics changed")

        events_path = session / "host2/events.jsonl"
        controller_counts_path = session / "host2/cost_controller_counts.json"
        controller_counts = read(controller_counts_path)
        if events_path.exists():
            public = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
            public = [row for row in public if row.get("event") in PUBLIC_EVENTS]
            events_sha256 = sha(events_path)
            event_source = {"path": str(events_path.relative_to(ROOT)), "exists": True,
                            "sha256": events_sha256}
        else:
            # A no-event online run may not create events.jsonl. Record its
            # absence only when independent run outputs also show no trigger,
            # request, child, or PROFILE epoch; do not invent a file/hash.
            if (controller_counts.get("natural_candidates") != 0
                    or controller_counts.get("manual_requests_sent") != []
                    or controller_counts.get("ignored_child_candidates") != 0
                    or any(row.get("profile_epoch", 0) != 0 for row in train)):
                raise ValueError("events.jsonl missing despite public event/profile evidence")
            online_log = session / "host2/online.log"
            if not online_log.exists() or online_log.stat().st_size != 0:
                raise ValueError("events.jsonl absent without an empty online.log cross-check")
            public = []
            events_sha256 = None
            event_source = {"path": str(events_path.relative_to(ROOT)), "exists": False,
                            "online_log_sha256": sha(online_log), "online_log_bytes": 0,
                            "absence_crosschecks": {"natural_candidates": 0,
                                "manual_requests_sent": [], "ignored_child_candidates": 0,
                                "profile_epoch_nonzero_steps": 0}}
        pred_path = HERE / f"recovery02_{layout}_predictions_without_truth.json"
        public_snapshot = {
            "protocol": "monitor_gc_268_infrastructure_recovery_public_v1",
            "layout": layout,
            "session_id": recovery["session_id"],
            "prediction_before_truth": True,
            "plan_sha256": sha(HERE / f"recovery02_{layout}_plan.json"),
            "source_freeze_sha256": sha(HERE / "freeze.json"),
            "events_sha256": events_sha256,
            "event_source": event_source,
            "events": public,
        }
        pred_path.write_text(json.dumps(public_snapshot, indent=2, sort_keys=True) + "\n")
        # The recovered node must have persisted the exact plan that passed the
        # configure_plan/node_cost CPU gate, including fixed-model opt-in.
        for host in ("host1", "host2"):
            actual_plan = read(session / host / "plan.json")
            if actual_plan != recovery:
                raise ValueError(f"{host}: recovered node plan differs from the frozen recovery plan")
            manifest_name = "node1_manifest.json" if host == "host1" else "node0_manifest.json"
            manifest = read(session / host / manifest_name)
            cfg = manifest.get("configuration", {})
            if cfg.get("startup_gc_fixed_model") is not True or cfg.get("startup_gc_policy") != recovery["startup_gc_policy"]:
                raise ValueError(f"{host}: manifest did not attest startup GC opt-in")
        completed = {
            "status": "complete",
            "layout": layout,
            "session_id": recovery["session_id"],
            "session_path": str(session.relative_to(ROOT)),
            "plan_sha256": sha(HERE / f"recovery02_{layout}_plan.json"),
            "prediction_sha256": sha(pred_path),
            "events_sha256": events_sha256,
            "events_file_absent": not events_path.exists(),
            "train_rows": len(train),
            "source_freeze_sha256": sha(HERE / "freeze.json"),
            "prediction_before_truth": True,
        }
        save(HERE / f"recovery02_complete_{layout}.json", completed)
        # Correct a stale owner label written by the reused detached helper without deleting its log.
        active_path = HERE / "active.json"
        if active_path.exists():
            active = read(active_path)
            active["status"] = "completed"
            active["gpu_owner"] = "extended64_protocol"
            save(active_path, active)
        save(HERE / f"recovery02_owner_metadata_correction_{layout}.json", {
            "recovery_session_id": recovery["session_id"],
            "actual_gpu_owner": "extended64_protocol",
            "stale_helper_label": "layer_campaign",
            "underlying_training_or_plan_changed": False,
        })
    except BaseException as error:
        save(HERE / f"recovery02_failure_{layout}.json", {
            "status": "failed_no_retry",
            "layout": layout,
            "session_id": recovery["session_id"],
            "error": repr(error),
            "active_session": owner.active,
            "no_automatic_retry": True,
        })
        if owner.active:
            for host, container, sudo in owner.NODES:
                try:
                    owner.rpc(host, owner.dock(host, container, sudo, ["touch", f"results/data16_sessions/{owner.active}/STOP"]))
                except Exception:
                    pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan, _, _, _ = preflight(args.layout)
    if args.execute:
        execute(args.layout)
    else:
        print(json.dumps({"status": "CPU_recovery_preflight_passed_no_GPU", "layout": args.layout,
                          "session_id": plan["session_id"], "plan_sha256": sha(HERE / f"recovery02_{args.layout}_plan.json")}, indent=2))


if __name__ == "__main__":
    main()
