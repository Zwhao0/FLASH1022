"""Root-owned short injection calibration with read-only host contention evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from flash_v4 import online_harness as h
from flash_v4.episode_qualification import qualify
from flash_v4.external_activity import vllm_activity
from flash_v4.frozen_localizer import verify
from flash_v4.health_centered_fusion import fingerprint, write_new
from flash_v4.online_driver import quiet_call
from flash_v4.online_evaluate import actuation_quality
from results.v4.preflight import online_r23_legacy_diagnostic_run as sensors


def pilot(row, truth, plan):
    case = row['case']
    quiet_call(h.launch, case)
    deadline = time.monotonic() + row.get('timeout', 180) + 40
    next_sample = 0
    active = []
    telemetry_ok = True
    with (h.LOCAL / f'telemetry_{case}.jsonl').open('x', buffering=1) as log, ThreadPoolExecutor(4) as pool:
        while time.monotonic() < deadline:
            states = json.loads(quiet_call(h.status, case))
            if time.monotonic() >= next_sample:
                futures = [pool.submit(sensors._remote_sample, node, target)
                           for target in ('host', 'container') for node in (0, 1)]
                records = [f.result() for f in futures]
                log.write(json.dumps(dict(records=records, local_wall_time=time.time())) + '\n')
                telemetry_ok = telemetry_ok and all(not sensors._sample_failed(r) for r in records)
                for r in records:
                    if r['target'] == 'host':
                        activity = vllm_activity(r.get('apps', {}), r.get('pmon', {}))
                        if activity['activity_observed']:
                            active.append(dict(node=r['node'], time=r.get('time'), workers=activity['active_worker_rows']))
                next_sample = time.monotonic() + 6
            if all(s.get('status') in ('completed', 'failed', 'timeout')
                   and s.get('private_mps_stopped') is True for s in states):
                quiet_call(h.fetch, case, 'final')
                root = h.LOCAL / case / 'final'
                train = root / 'node0/training/train.jsonl'
                rows = [json.loads(s) for s in train.read_text().splitlines()] if train.exists() else []
                strength = qualify(rows, truth['window'], plan['strength_contract'])
                actor = actuation_quality(root, truth, row['steps'], budget=plan['actor_budget'])
                completed = all(s['status'] == 'completed' and s['returncode'] == 0 for s in states)
                if not completed or not actor['passed'] or not telemetry_ok:
                    strength['qualified'] = False
                    strength['reasons'].append('runtime_actuation_or_telemetry_failed')
                measured = [r['walltime'] for r in rows if
                    truth['window']['first'] - plan['strength_contract']['baseline_steps'] <= r['iteration'] <= truth['window']['last']]
                concurrent = [r for r in active if measured and min(measured) <= r['time'] <= max(measured)]
                if concurrent:
                    strength['qualified'] = False
                    strength['reasons'].append('observed_external_compute_contention')
                result = dict(case=case, topology=row['topology'], target_rank=truth['target_rank'],
                    requested_ratio=truth['ratio'], nodes_completed=completed, actor=actor,
                    strength=strength, external_active_samples=active, telemetry_ok=telemetry_ok,
                    external_activity_during_measurement=bool(concurrent),
                    cleanup_confirmed=True, training=fingerprint(train) if train.exists() else None)
                write_new(h.LOCAL / (case + '_pilot_result.json'), result)
                print(json.dumps(dict(case=case, pilot_completed=completed,
                    actor_passed=actor['passed'], qualified=strength['qualified'],
                    reasons=strength['reasons'], metrics=strength.get('metrics'),
                    external_active_samples=len(active)), ensure_ascii=False), flush=True)
                return result
            time.sleep(2)
    write_new(h.LOCAL / (case + '_cleanup_unconfirmed.json'), dict(states=states, no_more_dispatch=True))
    raise RuntimeError('pilot terminal cleanup unconfirmed; no more dispatch')


def run(path):
    plan = json.loads(path.read_text()); plan_hash = fingerprint(path)
    verify(plan['source_files'] + plan['training_sources_before'])
    private = Path(plan['private_execution'])
    if fingerprint(private) != plan['private_execution_receipt']:
        raise ValueError('private plan changed')
    truth = json.loads(private.read_text())['cases']
    h.configure_plan(path); sensors.PLAN = path
    if h.LOCAL.exists():
        raise FileExistsError('no pilot release overwrite or automatic retry')
    if not sensors.precheck(plan, 'short_pilot_before_stage'):
        return 3
    h.stage(); h.build_actor()
    results = []
    for row in plan['development_GPU_sessions']:
        if row['mode'] != 'off' or row['steps'] != 48:
            raise ValueError('pilot must be48-step actuation-only calibration')
        if not sensors.precheck(plan, 'short_pilot_' + row['case']):
            write_new(h.LOCAL / 'batch_stopped.json', dict(reason='busy_environment', completed=len(results)))
            return 3
        verify(plan['source_files'] + plan['training_sources_before'])
        if fingerprint(path) != plan_hash:
            raise ValueError('public pilot plan changed')
        result = pilot(row, truth[row['case']], plan)
        results.append(result)
        if not result['nodes_completed'] or not result['actor']['passed'] or not result['telemetry_ok']:
            write_new(h.LOCAL / 'batch_failed.json', dict(case=row['case'], no_more_dispatch=True))
            return 1
    write_new(h.LOCAL / 'pilot_qualification_private.json', dict(results=results,
        all_qualified=all(r['strength']['qualified'] for r in results), plan=plan_hash,
        source=fingerprint(Path(__file__)), accuracy_claimed=False,
        scope='8-step actual slowdown calibration before separate32-step automatic tests'))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().plan))
