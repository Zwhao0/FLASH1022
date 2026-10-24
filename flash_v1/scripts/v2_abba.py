"""Serial, explicit v2 ABBA execution; retained IDs are never rerun."""
import argparse
import hashlib
import json
from pathlib import Path

from flash_v1.scripts.data16 import ROOT, run
from flash_v1.scripts.v2_benchmark import benchmark_plan, score_group
from flash_v1.scripts.v2_report import score_online


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',choices=('legacy','node_local_pp'),required=True)
    parser.add_argument('--family',choices=('profile_healthy','profile','monitor','full_online'),required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--backend',choices=('native_semantic','semantic_only'),default='native_semantic')
    parser.add_argument('--model',default='results/v2_collection/candidate_model.json')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--start-at',type=int,default=0,choices=range(4))
    args=parser.parse_args()
    group=f'{args.family}_{args.layout}_{args.tag}'
    output=ROOT/'results/v2_benchmarks'/group
    output.mkdir(parents=True,exist_ok=True)
    labels=('a1','b1','b2','a2');modes=('A','B','B','A')
    plans=[]
    for label,mode in zip(labels,modes):
        session=f'v2_ab_{group}_{label}'
        p=benchmark_plan(session,args.layout,args.seed,mode,args.family,args.model,args.backend)
        if (args.family.startswith('profile') or args.family=='full_online') and label!='a1':
            p['benchmark_work_reference_session']=f'v2_ab_{group}_a1'
        plans.append(p)
    protocol=output/'protocol.json'
    body=json.dumps(dict(order=list(labels),plans=plans,
        analysis='Exclude only first12 warmup; all strata, adjacent pairs, numerical identities and physical injection traces retained.'),indent=2)+'\n'
    if protocol.exists() and protocol.read_text()!=body:
        raise ValueError('existing protocol differs; use a fresh tag')
    if not protocol.exists():protocol.write_text(body)
    paths=[ROOT/'results/data16_sessions'/p['session_id'] for p in plans]
    active=ROOT/'results/v2_benchmarks/active.json'
    if args.run:
        for index,(p,path) in enumerate(zip(plans,paths)):
            if index<args.start_at:
                for node,host in enumerate(('host2','host1')):
                    m=json.loads((path/host/f'node{node}_manifest.json').read_text())
                    if m['status']!='completed':raise ValueError('cannot skip incomplete node')
                continue
            active.write_text(json.dumps(dict(group=group,session=p['session_id'],status='running',index=index),indent=2)+'\n')
            print('START',p['session_id'],flush=True)
            try:
                run(p,timeout=1200)
            except BaseException as error:
                active.write_text(json.dumps(dict(group=group,session=p['session_id'],status='failed',error=repr(error),index=index),indent=2)+'\n')
                raise
            print('COMPLETE',p['session_id'],flush=True)
            if args.family=='full_online' and p['benchmark_mode']=='B':
                validation=score_online(path)
                if (not validation['integrity_passed'] or validation['correct']!=validation['denominator']
                        or validation['healthy_detection_count']):
                    active.write_text(json.dumps(dict(group=group,session=p['session_id'],status='failed_online_validation'),indent=2)+'\n')
                    raise RuntimeError('full online validation failed; retained session must be reviewed')
    report=score_group(paths,output/'report.json')
    report['scorer_sha256']=hashlib.sha256((ROOT/'flash_v1/scripts/v2_benchmark.py').read_bytes()).hexdigest()
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    active.write_text(json.dumps(dict(group=group,status='scored',report=str(output/'report.json'),
        integrity_passed=report['integrity_passed'],target_point_estimate_met=report['target_point_estimate_met']),indent=2)+'\n')
    print(json.dumps(dict(group=group,integrity=report['integrity_passed'],target=report['target_point_estimate_met'],
        overhead={k:v['overhead_pct'] for k,v in report['strata'].items()})),flush=True)
    if not report['integrity_passed']:raise SystemExit(2)
    if not report['target_point_estimate_met']:raise SystemExit(3)


if __name__=='__main__':
    main()
