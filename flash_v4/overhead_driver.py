"""Fixed mirrored health measurements. One owner; all failures retained."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

from flash_v4 import online_harness as h
from flash_v4.online_driver import quiet_call, quality
from flash_v4.health_centered_fusion import fingerprint, write_new


def run(plan_path):
    h.configure_plan(plan_path)
    plan = json.loads(Path(plan_path).read_text())
    root = h.LOCAL / plan['measurement_id']
    root.mkdir(exist_ok=False)
    write_new(root/'receipt.json',{'plan':fingerprint(Path(plan_path)),
              'driver':fingerprint(Path(__file__)),'harness':fingerprint(Path(h.__file__)),
              'owner':'root','start':time.time()})
    results=[]
    with (root/'progress.jsonl').open('x',buffering=1) as log:
        for row in plan['development_GPU_sessions']:
            case=row['case'];quiet_call(h.launch,case);deadline=time.monotonic()+220
            live=False;error=None
            while time.monotonic()<deadline:
                states=json.loads(quiet_call(h.status,case))
                log.write(json.dumps({'case':case,'time':time.time(),'nodes':states})+'\n')
                if all(s.get('profile_ready_ranks')==8 for s in states) and not live and error is None:
                    try:
                        quiet_call(h.fetch,case,'live');quiet_call(h.localize,case,'live');live=True
                    except Exception as exc:
                        error=type(exc).__name__+': '+str(exc)
                if all(s.get('status') in ('completed','failed','timeout') and s.get('private_mps_stopped') is True for s in states):
                    quiet_call(h.fetch,case,'final');break
                time.sleep(3)
            else:
                write_new(root/'cleanup_unconfirmed.json',{'case':case,'states':states})
                return 2
            folder=h.LOCAL/case
            records=[json.loads(x) for x in (folder/'final/node0/training/train.jsonl').read_text().splitlines()]
            measured=[r for r in records if not r['warmup']]
            profile=[r['wall_iteration_ms'] for r in records if 36<=r['iteration']<=43]
            result={'case':case,'category':row['category'],'topology':row['topology'],
                    'seed':row['seed'],'steps':len(measured),
                    'all_step_median_ms':median(r['wall_iteration_ms'] for r in measured),
                    'all_step_mean_ms':sum(r['wall_iteration_ms'] for r in measured)/len(measured),
                    'profile_window_median_ms':median(profile),
                    'healthy_duration_seconds':sum(r['wall_iteration_ms'] for r in measured)/1000,
                    'profile_triggered':live,'error':error,
                    'nodes_completed':all(s['status']=='completed' and s['returncode']==0 for s in states)}
            if live:
                result['profile_quality']=quality(case)
            write_new(folder/'measurement.json',result);results.append(result)
            (root/'state.json').write_text(json.dumps({'completed':len(results),'last':result},indent=2))
            if not result['nodes_completed'] or error or (live and not result['profile_quality']['quality_passed']):
                write_new(root/'stopped.json',{'case':case,'reason':'retained training/profile failure; no more dispatch'})
                return 1
        baseline=[r for r in results if r['category']=='off']
        summary={'runs':results,'scope':'paired point estimates; no confidence or population FPR guarantee'}
        for category in ('monitor','profile'):
            values=[r for r in results if r['category']==category]
            key='all_step_mean_ms' if category=='monitor' else 'profile_window_median_ms'
            base=median(r[key] for r in baseline)
            summary[category]={'baseline_ms':base,'measured_ms':median(r[key] for r in values),
                               'overhead_fraction_point':median(r[key] for r in values)/base-1,
                               'run_values_ms':[r[key] for r in values]}
        write_new(root/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('monitor','profile')}))
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    raise SystemExit(run(p.parse_args().plan))
