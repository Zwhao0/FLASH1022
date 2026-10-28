"""One finite, automatically supervised runtime integration case. No retries."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time

from flash_v4 import online_harness as harness
from flash_v4.health_centered_fusion import fingerprint, write_new


def quiet_call(function, *args):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function(*args)
    return buffer.getvalue()


def control_integrity(summaries, ready, endings, requests, manual=False):
    """Check complete rank identities and actual monitor evidence, without labels."""
    ranks = set(range(16))
    def complete(rows):
        return len(rows) == 16 and {r.get('rank') for r in rows} == ranks
    if not complete(summaries) or not complete(endings):
        return False
    requested = bool(ready)
    if any(e.get('profile_requested') is not requested for e in endings):
        return False
    for end in endings:
        expected = ('manual_integration' if manual else
                    'completed' if end['rank'] == 0 else 'peer_completed')
        if end.get('monitor_status') != expected:
            return False
        if not manual and end['rank'] == 0 and end.get('worker_returncode') != 0:
            return False
    if not requested:
        return not requests and not manual
    if (not complete(ready) or len({(r['first'], r['last']) for r in ready}) != 1
            or any(not e.get('profile_complete') for e in endings)):
        return False
    first, last = ready[0]['first'], ready[0]['last']
    if last - first != 7 or first < 4 or (first - 4) % 8:
        return False
    if manual:
        return not requests
    return (len(requests) == 1 and requests[0].get('method') == 'flash_v1_Rbeast'
            and type(requests[0].get('iteration')) is int
            and 0 < requests[0]['iteration'] < first)


def quality(case):
    root = harness.LOCAL / case
    files = sorted((root / 'final').glob('node*/observations/rank_*.jsonl'))
    summaries = [json.loads(p.read_text().splitlines()[-1]) for p in files]
    ready = [json.loads(p.read_text()) for p in (root / 'final').glob('node*/control/ready_rank*.json')]
    endings=[]
    for p in (root/'final').glob('node*/control/online_rank*.jsonl'):
        endings.extend(r for r in [json.loads(s) for s in p.read_text().splitlines()]
                       if r.get('kind')=='monitor_end')
    row = harness.case_row(case)
    requests = [json.loads(p.read_text()) for p in
                (root / 'final').glob('node*/control/profile_request.json')]
    good = (len(files) == len(ready) == 16 and all(s.get('window_complete') is True
            and s.get('completed_profile_iterations') == 8 and s.get('dropped') == 0
            and s.get('skipped') == 0 for s in summaries)
            and max(p.stat().st_size for p in files) <= 1048576
            and len({(r['first'],r['last']) for r in ready}) == 1)
    if not ready:
        good=(len(files)==16 and all(s.get('emitted')==0 and s.get('stop_reason')=='profile_not_started' for s in summaries)
              and len(endings)==16 and all(e.get('monitor_status') in ('completed','peer_completed')
                                          and not e.get('profile_requested') for e in endings))
    integrity = control_integrity(summaries, ready, endings, requests,
                                  manual=bool(row.get('force_profile_at')))
    good = good and integrity
    prediction_path = root / 'live/frozen_prediction.json'
    prediction = json.loads(prediction_path.read_text()) if prediction_path.exists() else {}
    training = root / 'final/node0/training/train.jsonl'
    last = json.loads(training.read_text().splitlines()[-1])
    result = {'case': case, 'quality_passed': good, 'rank_files': len(files),
              'complete_ranks': sum(s.get('window_complete') is True for s in summaries),
              'profile_triggered':bool(ready),'monitor_end_records':len(endings),
              'control_integrity_passed': integrity,
              'automatic_profile_completed': bool(good and ready and not row.get('force_profile_at')),
              'max_rank_bytes': max((p.stat().st_size for p in files), default=0),
              'window': [ready[0]['first'],ready[0]['last']] if ready else None,
              'alarm_rank': prediction.get('alarm_rank'),
              'prediction_during_training': bool(prediction and prediction['prediction_wall_time'] < last['walltime']),
              'manual_integration_only': bool(row.get('force_profile_at')),
              'automatic_detection_success_claimed': False,
              'training_end_rank0_wall_time': last['walltime']}
    write_new(root / 'integration_quality.json', result)
    return result


def read_status(case, log):
    """Retry one failed read-only SSH status observation, never GPU dispatch."""
    try:
        return json.loads(quiet_call(harness.status, case))
    except RuntimeError as error:
        if 'banner exchange' not in str(error):
            raise
        log.write(json.dumps({'kind': 'status_observation_retry',
                              'reason': 'SSH_banner_exchange_failure', 'attempt': 2}) + '\n')
        time.sleep(.5)
        return json.loads(quiet_call(harness.status, case))


def execute(case):
    marker = harness.LOCAL / ('driver_' + case)
    marker.mkdir(exist_ok=False)
    write_new(marker / 'controller_receipt.json', {'source':fingerprint(Path(__file__)),
               'harness':fingerprint(Path(harness.__file__)), 'start':time.time(),
               'owner':'root','case':harness.case_row(case),'max_seconds':220})
    quiet_call(harness.launch,case)
    deadline = time.monotonic() + 220
    localized, error = False, None
    with (marker / 'progress.jsonl').open('x',buffering=1) as log:
        while time.monotonic() < deadline:
            states = read_status(case, log)
            log.write(json.dumps({'wall_time':time.time(),'nodes':states})+'\n')
            if all(s.get('profile_ready_ranks')==8 for s in states) and not localized and error is None:
                try:
                    quiet_call(harness.fetch,case,'live')
                    quiet_call(harness.localize,case,'live')
                    localized = True
                except Exception as exc:
                    error = type(exc).__name__+': '+str(exc)
                    log.write(json.dumps({'error':error})+'\n')
            if all(s.get('status') in ('completed','failed','timeout')
                   and s.get('private_mps_stopped') is True for s in states):
                quiet_call(harness.fetch,case,'final')
                try:
                    result = quality(case)
                except Exception as exc:
                    result = {'case': case, 'quality_passed': False,
                              'quality_error': type(exc).__name__ + ': ' + str(exc)}
                result['nodes_completed'] = all(s['status']=='completed' and s['returncode']==0 for s in states)
                result['error'] = error
                write_new(marker/'result.json',result)
                print(json.dumps(result))
                return 0 if result['quality_passed'] and result['nodes_completed'] and not error else 1
            time.sleep(3)
    write_new(marker/'unconfirmed_cleanup.json',{'error':'terminal status/cleanup not confirmed; no more dispatch','last_nodes':states})
    return 2


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--plan')
    args=p.parse_args()
    if args.plan:harness.configure_plan(args.plan)
    raise SystemExit(execute(args.case))
