"""Run the fixed eight cases in an isolated staged-production container root."""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
PROD = ROOT / "results/layer_campaign/stage_forward_v1/sparse3_gc_production_draft"
STAGED_TAR = HERE / "staged_flash_v1.tar"
STAGED_TREE = HERE / "staged_tree.json"
EXPERIMENT_ROOT = "/workspace/FLASH1022/tmp/sparse3_gcdiag_reg8"
LAYOUTS = ("legacy", "node_local_pp")
ALLOWED = {"detection", "profile_schedule", "localization"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_local_contract():
    ready = read(HERE / "ready.json")
    freeze_path = HERE / "source_freeze.json"
    freeze = read(freeze_path)
    design_path = HERE / "design.json"
    design = read(design_path)
    if ready.get("status") != "ready_for_new_fixed_case_sparse3_gcdiag_regression" or ready.get("fixed_denominator") != 8:
        raise ValueError("fixed reg8 ready contract mismatch")
    if ready.get("source_freeze_sha256") != sha(freeze_path) or design.get("source_freeze", {}).get("sha256") != sha(freeze_path):
        raise ValueError("candidate source freeze binding mismatch")
    for name, expected in ready.get("artifacts", {}).items():
        path = freeze_path if name == "source_freeze.json" else HERE / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"ready artifact mismatch: {name}")

    old_dir = ROOT / "results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_reverted_baseline"
    old_design = read(old_dir / "design.json")
    if design.get("training_seed") != old_design.get("training_seed") or design.get("layouts") != old_design.get("layouts"):
        raise ValueError("case design/seed differs from the fixed baseline cases")
    if design.get("fixed_denominator") != 8:
        raise ValueError("candidate denominator must remain eight")
    for layout in LAYOUTS:
        plan = read(HERE / f"formal_{layout}_plan.json")
        parent = read(old_dir / f"formal_{layout}_plan.json")
        changed = {key for key in set(plan) | set(parent) if plan.get(key) != parent.get(key)}
        if changed != {"session_id", "freeze_manifest_sha256", "freeze_manifest_path", "stage_forward_aux_sampling", "gc_callbacks_diagnostic", "monitor_cost_timing"}:
            raise ValueError(f"unexpected fixed plan changes for {layout}: {sorted(changed)}")
        if plan.get("steps") != 305 or plan.get("seed") != 2032905414 or plan.get("parallel_layout") != layout:
            raise ValueError(f"plan identity mismatch: {layout}")
        if plan.get("freeze_manifest_sha256") != sha(freeze_path) or plan.get("freeze_manifest_path") != str(freeze_path.relative_to(ROOT)):
            raise ValueError(f"plan source binding mismatch: {layout}")
        expected_sampling = {
            "schema": freeze.get("sampling_schema"), "window_size": 8,
            "sample_offsets": freeze.get("sample_offsets"),
            "expected_aux_rows": freeze.get("expected_aux_rows_per_window"),
            "core_evidence_steps": 8,
        }
        if (plan.get("stage_forward_aux_sampling") != expected_sampling
                or plan.get("gc_callbacks_diagnostic") is not True
                or plan.get("monitor_cost_timing") is not True):
            raise ValueError(f"sparse3/GC diagnostic flags mismatch: {layout}")
        expected_sampling = {
            "schema": freeze.get("sampling_schema"),
            "window_size": 8,
            "sample_offsets": freeze.get("sample_offsets"),
            "expected_aux_rows": freeze.get("expected_aux_rows_per_window"),
            "core_evidence_steps": 8,
        }
        if (plan.get("stage_forward_aux_sampling") != expected_sampling
                or plan.get("gc_callbacks_diagnostic") is not True
                or plan.get("monitor_cost_timing") is not True):
            raise ValueError(f"sparse3/GC diagnostic plan flags mismatch: {layout}")
        if [row for row in plan["intervals"] if row.get("segment") == "fault"] != [row for row in parent["intervals"] if row.get("segment") == "fault"]:
            raise ValueError(f"targets/doses/case order changed: {layout}")

    source_files = freeze.get("files", {})
    binding = read(PROD / "source_binding.json")
    production_root = HERE / "production"
    diag_ready = read(ROOT / "results/layer_campaign/stage_forward_v1/sparse3_gc_diag_draft/ready.json")
    overlay_files = diag_ready.get("files", {})
    for name, record in binding.get("production_files", {}).items():
        expected = overlay_files.get(name, {}).get("after_sha256", record.get("after_sha256"))
        if source_files.get(name) != expected or sha(production_root / name) != expected:
            raise ValueError(f"production staged file not source-pinned: {name}")
    for name, record in overlay_files.items():
        expected = record.get("after_sha256")
        if source_files.get(name) != expected or sha(production_root / name) != expected:
            raise ValueError(f"diagnostic overlay file not source-pinned: {name}")
    if sha(STAGED_TAR) != read(STAGED_TREE).get("tar_sha256"):
        raise ValueError("staged flash_v1 tar changed")
    tree = read(STAGED_TREE)
    with tarfile.open(STAGED_TAR, "r") as archive:
        entries = {member.name: member for member in archive.getmembers()}
        for name, expected in tree.get("files", {}).items():
            member = entries.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"staged tar missing file: {name}")
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if digest != expected:
                raise ValueError(f"staged tar file hash mismatch: {name}")
    return ready, freeze, freeze_path, design


def remote_code_for_install(experiment_root: str, file_map: dict, binary_map: dict):
    expected = {**file_map, **binary_map}
    return "\n".join([
        "import base64,hashlib,io,json,sys,tarfile",
        "from pathlib import Path",
        f"root=Path({experiment_root!r})",
        "assert not root.exists(), 'candidate experiment root already exists; preserve it'",
        "root.mkdir(parents=True)",
        "data=base64.b64decode(sys.stdin.read())",
        "archive=tarfile.open(fileobj=io.BytesIO(data),mode='r:')",
        "members=archive.getmembers()",
        "assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)",
        "archive.extractall(root)",
        "archive.close()",
        "links={'Greyhound-master':'/workspace/FLASH1022/Greyhound-master','models':'/workspace/FLASH1022/models','results':'/workspace/FLASH1022/results','data16_plans':'/workspace/FLASH1022/data16_plans','flash_v1/build':'/workspace/FLASH1022/flash_v1/build'}",
        "for name,target in links.items():",
        " p=root/name; t=Path(target); assert t.exists(), ('missing external runtime asset',name,str(t)); p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists() and not p.is_symlink(); p.symlink_to(t,target_is_directory=True)",
        f"expected=json.loads({json.dumps(json.dumps(expected,sort_keys=True))})",
        "actual={}",
        "for name in expected:",
        " p=root/name",
        " assert p.is_file(), ('missing pinned candidate file',name,str(p))",
        " actual[name]=hashlib.sha256(p.read_bytes()).hexdigest()",
        "assert actual==expected, {k:(expected.get(k),actual.get(k)) for k in expected if expected.get(k)!=actual.get(k)}",
        "print(json.dumps({'status':'candidate_overlay_ready','root':str(root),'files_verified':len(actual),'source_sha256':actual},sort_keys=True))",
    ])


def deploy_overlay(freeze, staged_tar):
    sys.path.insert(0, str(ROOT))
    from results.layer_campaign.continuation_v2.gates import owned_runner as r
    code = remote_code_for_install(EXPERIMENT_ROOT, freeze["files"], freeze.get("binary_files", {}))
    payload = base64.b64encode(staged_tar.read_bytes()).decode()
    checks = {}
    for host, container, sudo in r.NODES:
        result = r.rpc(host, r.dock(host, container, sudo, ["python", "-c", code]), input=payload)
        checks[host] = json.loads(result)
    if any(row.get("status") != "candidate_overlay_ready" for row in checks.values()):
        raise RuntimeError("candidate source deployment did not verify")
    if checks["host1"]["source_sha256"] != checks["host2"]["source_sha256"]:
        raise RuntimeError("candidate source differs across hosts")
    atomic_json(HERE / "remote_candidate_source_check.json", checks)
    return checks


def start_resource_sidecars():
    """Start low-rate read-only GPU/process snapshots inside each owned container."""
    sys.path.insert(0, str(ROOT))
    from results.layer_campaign.continuation_v2.gates import owned_runner as r
    sidecar = r"""
import datetime,json,os,subprocess,sys,time
marker='sparse3-reg8-resource-sidecar'
log='/tmp/sparse3_gcdiag_reg8_resource.jsonl'
def capture(argv):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=6)
  return {'returncode':p.returncode,'stdout':p.stdout[-20000:],'stderr':p.stderr[-2000:]}
 except Exception as e:return {'error':repr(e)}
while True:
 t=time.monotonic()
 row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':marker,
  'gpu':capture(['nvidia-smi','--query-gpu=timestamp,index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits']),
  'pmon':capture(['nvidia-smi','pmon','-c','1','-s','um']),
  'compute_processes':capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
  'cpu_processes':capture(['ps','-eo','pid,pcpu,psr,comm,args','--sort=-pcpu']),
  'loadavg':open('/proc/loadavg').read().strip() if os.path.exists('/proc/loadavg') else None}
  try:
   row['cgroup']={name:open(path).read().strip() for name,path in {'cpu.stat':'/sys/fs/cgroup/cpu.stat','cpuset.cpus.effective':'/sys/fs/cgroup/cpuset.cpus.effective','memory.current':'/sys/fs/cgroup/memory.current','memory.max':'/sys/fs/cgroup/memory.max'}.items() if os.path.exists(path)}
  except Exception as e:row['cgroup_error']=repr(e)
  with open(log,'a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
  time.sleep(max(1,10-(time.monotonic()-t)))
"""
    results = {}
    for host, container, sudo in r.NODES:
        code = "import json,subprocess,sys;from pathlib import Path;Path('/tmp/sparse3_gcdiag_reg8_resource.jsonl').unlink(missing_ok=True);Path('/tmp/sparse3_gcdiag_reg8_resource.pid').unlink(missing_ok=True);script=" + repr(sidecar) + ";p=subprocess.Popen([sys.executable,'-u','-c',script],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);Path('/tmp/sparse3_gcdiag_reg8_resource.pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid,'log':'/tmp/sparse3_gcdiag_reg8_resource.jsonl','host':" + repr(host) + "}))"
        results[host] = json.loads(r.rpc(host, r.dock(host, container, sudo, ["python", "-c", code])))
    atomic_json(HERE / "resource_sidecars_started.json", results)
    return results


def stop_resource_sidecars():
    """Stop only our marked sidecar and preserve its read-only log locally."""
    sys.path.insert(0, str(ROOT))
    from results.layer_campaign.continuation_v2.gates import owned_runner as r
    results = {}
    for host, container, sudo in r.NODES:
        code = r"""import json,os,signal,time
from pathlib import Path
pidfile=Path('/tmp/sparse3_gcdiag_reg8_resource.pid');log=Path('/tmp/sparse3_gcdiag_reg8_resource.jsonl');status='missing'
if pidfile.exists():
 pid=int(pidfile.read_text());cmd=Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and b'sparse3-reg8-resource-sidecar' in cmd.read_bytes():
  os.kill(pid,signal.SIGTERM);status='stopped_owned';time.sleep(.2)
 else:status='pid_not_owned_or_exited'
text=log.read_text() if log.exists() else ''
print(json.dumps({'status':status,'pid':pid if pidfile.exists() else None,'log':text}))
"""
        results[host] = json.loads(r.rpc(host, r.dock(host, container, sudo, ["python", "-c", code])))
        (HERE / f"resource_sidecar_{host}.jsonl").write_text(results[host].get("log", ""))
    atomic_json(HERE / "resource_sidecars_stopped.json", {
        host: {"status": row.get("status"), "pid": row.get("pid"),
               "records": len([line for line in row.get("log", "").splitlines() if line.strip()]),
               "log_sha256": sha(HERE / f"resource_sidecar_{host}.jsonl")}
        for host, row in results.items()
    })


def remove_overlay():
    sys.path.insert(0, str(ROOT))
    from results.layer_campaign.continuation_v2.gates import owned_runner as r
    code = f"from pathlib import Path;p=Path({EXPERIMENT_ROOT!r});assert p.exists() and p.is_dir();import shutil;shutil.rmtree(p);print('removed isolated candidate root')"
    results = {}
    for host, container, sudo in r.NODES:
        results[host] = r.rpc(host, r.dock(host, container, sudo, ["python", "-c", code])).strip()
    atomic_json(HERE / "isolated_overlay_cleanup.json", {"status": "removed", "hosts": results})


def session_complete(session: Path, plan: dict):
    status = {}
    for host, manifest_name, train_name in (
        ("host2", "node0_manifest.json", "train.jsonl"),
        ("host1", "node1_manifest.json", None),
    ):
        manifest = read(session / host / manifest_name)
        train = ([json.loads(line) for line in (session / host / train_name).read_text().splitlines() if line.strip()]
                 if train_name else [])
        if manifest.get("status") != "completed" or (train_name and (len(train) != 305 or int(train[-1].get("iteration", -1)) != 305)):
            raise RuntimeError(f"incomplete candidate session {session.name}/{host}")
        config = manifest.get("configuration", {})
        if config.get("startup_gc_fixed_model") is not True or config.get("startup_gc_policy", {}).get("schema") != "startup_process_lifetime_heap_freeze_v2":
            raise RuntimeError(f"startup GC fixed-model opt-in missing in {session.name}/{host}")
        if (config.get("stage_forward_aux_sampling") != plan.get("stage_forward_aux_sampling")
                or config.get("gc_callbacks_diagnostic") is not True):
            raise RuntimeError(f"sparse3/diagnostic configuration missing in {session.name}/{host}")
        persisted_plan = read(session / host / "plan.json")
        sys.path.insert(0, str(ROOT / "results/layer_campaign/stage_forward_v1/sparse3_gc_production_draft/production"))
        from flash_v1.startup_gc import configure_plan
        if persisted_plan != configure_plan(plan):
            raise RuntimeError(f"persisted configured plan differs in {session.name}/{host}")
        gc_paths = sorted((session / host).glob("gc_iteration_diagnostic_rank*.json"))
        if len(gc_paths) != 8:
            raise RuntimeError(f"expected 8 rank-local GC diagnostic files in {session.name}/{host}")
        gc_records = [read(path) for path in gc_paths]
        if any(record.get("schema") != "whole_iteration_gc_diagnostic_v1"
               or record.get("status") != "complete"
               or record.get("no_GPU_sync_or_GC_policy_change") is not True
               for record in gc_records):
            raise RuntimeError(f"GC diagnostic incomplete/out of contract in {session.name}/{host}")
        cost_paths = sorted((session / host).glob("monitor_cost_rank*.json"))
        if len(cost_paths) != 8:
            raise RuntimeError(f"expected 8 monitor-cost rank files in {session.name}/{host}")
        if any(len(read(path)) != 305 for path in cost_paths):
            raise RuntimeError(f"monitor-cost trace is incomplete in {session.name}/{host}")
        status[host] = {"manifest_sha256": sha(session / host / manifest_name), "train_rows": len(train) if train_name else None,
                        "last_iteration": train[-1].get("iteration") if train_name else None,
                        "startup_gc_fixed_model": config.get("startup_gc_fixed_model"),
                        "startup_gc_policy": config.get("startup_gc_policy"),
                        "gc_diagnostic_sha256": {path.name: sha(path) for path in gc_paths},
                        "gc_collections": sum(len(record.get("collections", [])) for record in gc_records),
                        "monitor_cost_rows": 305 * len(cost_paths)}
    return status


def main():
    if any((HERE / name).exists() for name in ("execution_start.json", "execution_complete.json", "execution_failure.json")):
        raise FileExistsError("candidate fixed-reg8 execution already exists; no automatic retry")
    ready, freeze, freeze_path, design = verify_local_contract()
    staged_entry = load_module("reg8_staged_entry", PROD / "staged_entry.py")
    staged_entry.check()
    deployment = deploy_overlay(freeze, STAGED_TAR)
    sidecars = start_resource_sidecars()
    start = {"status": "starting", "ready_sha256": sha(HERE / "ready.json"),
             "source_freeze_sha256": sha(freeze_path), "staged_tar_sha256": sha(STAGED_TAR),
             "design_sha256": sha(HERE / "design.json"), "GPU_owner": "extended64_protocol",
             "candidate_entry": "staged production data16.run(plan, startup_gc_fixed_model=True)",
             "experiment_root": EXPERIMENT_ROOT, "host_source_check": deployment,
             "resource_sidecars": sidecars,
             "fixed_denominator": 8, "no_automatic_retry": True}
    atomic_json(HERE / "execution_start.json", start)
    sessions = {}
    public_snapshots = {}
    try:
        for layout in LAYOUTS:
            plan = read(HERE / f"formal_{layout}_plan.json")
            session = staged_entry.run_staged(plan, EXPERIMENT_ROOT, timeout=7200)
            node_status = session_complete(session, plan)
            events_path = session / "host2/events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
            public = [row for row in events if row.get("event") in ALLOWED]
            prediction = {
                "protocol": "peer_priority_reg8_sparse3_gcdiag_public_snapshot_v1",
                "layout": layout,
                "session": session.name,
                "prediction_before_truth": True,
                "source_freeze_sha256": sha(freeze_path),
                "plan_sha256": sha(HERE / f"formal_{layout}_plan.json"),
                "public_events_sha256": sha(events_path),
                "events": public,
            }
            prediction_path = HERE / f"{layout}_predictions_without_truth.json"
            atomic_json(prediction_path, prediction)
            sessions[layout] = {
                "session": str(session.relative_to(ROOT)),
                "session_id": session.name,
                "plan_sha256": sha(HERE / f"formal_{layout}_plan.json"),
                "host_completion": node_status,
                "public_events_sha256": sha(events_path),
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "prediction_sha256": sha(prediction_path),
            }
            public_snapshots[layout] = {k: sessions[layout][k] for k in ("session_id", "plan_sha256", "public_events_sha256", "prediction_sha256")}
            atomic_json(HERE / "public_predictions_before_truth.json", {
                "protocol": "peer_priority_reg8_sparse3_gcdiag_prediction_freeze_v1",
                "prediction_before_truth": True,
                "fixed_denominator": 8,
                "source_freeze_sha256": sha(freeze_path),
                "layouts": public_snapshots,
            })
        atomic_json(HERE / "execution_complete.json", {
            "status": "complete",
            "fixed_denominator": 8,
            "steps_per_layout": 305,
            "source_freeze_sha256": sha(freeze_path),
            "prediction_before_truth": True,
            "sessions": sessions,
        })
    except BaseException as error:
        atomic_json(HERE / "execution_failure.json", {
            "error": repr(error),
            "sessions_completed": sessions,
            "no_automatic_retry": True,
            "overlay_preserved_for_audit": True,
        })
        raise
    finally:
        # The overlay is isolated from the shared source tree and is safe to remove
        # once the launcher has stopped all child processes.
        try:
            stop_resource_sidecars()
        finally:
            remove_overlay()

    print(json.dumps({"status": "complete", "fixed_denominator": 8,
                      "source_freeze_sha256": sha(freeze_path),
                      "prediction_freeze_sha256": sha(HERE / "public_predictions_before_truth.json"),
                      "sessions": {layout: row["session_id"] for layout, row in sessions.items()}}, indent=2))


if __name__ == "__main__":
    main()
