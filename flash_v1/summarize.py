"""Small trace summary: retain failures/inconclusive localization explicitly."""
import argparse
import json
from pathlib import Path
import statistics as s


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summary(path):
    manifest = json.loads((path/'manifest.json').read_text()) if (path/'manifest.json').exists() else {}
    train=[r for r in read(path/'train.jsonl') if not r['warmup']]
    events=read(path/'events.jsonl');injections=read(path/'injection.jsonl')
    profiles=read(path/'local.jsonl')
    intervals=[];start=None
    for event in profiles:
        if event['event']=='profile_start':start=event['wall_time']
        elif event['event']=='profile_end' and start is not None:intervals.append((start,event['wall_time']));start=None
    normal=[r for r in train if not any(a<=r['walltime']<=b for a,b in intervals)]
    def stats(rows):
        xs=[r['time_ms'] for r in rows]
        return dict(n=len(xs),mean_ms=s.mean(xs) if xs else None,median_ms=s.median(xs) if xs else None)
    result=dict(path=str(path),all_steps=stats(train),monitoring_steps=stats(normal),detections=[e for e in events if e['event']=='detection'],localizations=[e for e in events if e['event']=='localization'],profiles=profiles)
    result.update(status=manifest.get('status', 'unknown'), error=manifest.get('error'),
                  configuration={key: manifest.get(key) for key in (
                      'mode', 'gradient_backend', 'gpus', 'expert_parallel_size',
                      'microbatches', 'delay_placement', 'fault_rank', 'delay_ms', 'environment')})
    if manifest.get('mode') == 'fault' and manifest.get('fault_kind') == 'gpu-delay':
        start = manifest['inject_at']
        end = start + manifest['fault_steps']
        stable = [r for r in train if not any(a-.5 <= r['walltime'] <= b+.5 for a,b in intervals)]
        result['by_injection_iteration'] = dict(
            before=stats([r for r in train if r['step'] <= start]),
            fault_without_profile=stats([r for r in stable if start < r['step'] <= end]),
            recovery=stats([r for r in train if r['step'] > end]),
            excluded_profile_boundary_seconds=.5)
    result['profile_samples'] = {p.name: len(json.loads(p.read_text()))
                                 for p in sorted(path.glob('profile_*.json')) if 'diagnostics' not in p.name}
    lowered=next((e for e in injections if e['event']=='clock_lowered'),None)
    delay_events=[event for log in sorted(path.glob('fault_rank*.jsonl')) for event in read(log)]
    delayed=next((e for e in delay_events if e['event']=='compute_delay_start'),None)
    if delayed is not None:
        lowered=delayed
        result['simulation_calibration']=next((e for e in delay_events if e['event']=='compute_delay_calibration'),None)
    restored=next((e for e in injections if e['event'].startswith('clock_restored')),None)
    if delayed is not None:
        restored=next((e for e in delay_events if e['event']=='compute_delay_end'),None)
    if lowered:
        result['fault']=lowered
        result['normal_before_fault']=stats([r for r in normal if r['walltime']<lowered['walltime']])
        result['degraded_without_profile']=stats([r for r in normal if r['walltime']-r['time_ms']/1000>=lowered['walltime'] and (not restored or r['walltime']<restored['walltime'])])
        result['false_triggers_before_fault']=sum(e['wall_time']<lowered['walltime'] for e in result['detections'])
        result['correct_localizations']=sum(e['result']['rank']==lowered['rank'] for e in result['localizations'])
        result['detection_latency_seconds']=[e['wall_time']-lowered['walltime'] for e in result['detections'] if e['wall_time']>=lowered['walltime']]
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--output',required=True);args=p.parse_args()
    results=[summary(Path(path)) for path in args.paths]
    Path(args.output).write_text(json.dumps(results,indent=2))
    for row in results:
        print(row['path'], row['monitoring_steps'], 'detections',len(row['detections']), 'ranks',[e['result']['rank'] for e in row['localizations']])
