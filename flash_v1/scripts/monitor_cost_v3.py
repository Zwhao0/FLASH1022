"""Small, predeclared MONITOR ABBA; independent of frozen diagnosis results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from flash_v1.scripts.data16 import run, ROOT
from flash_v1.scripts.v2_benchmark import benchmark_plan

DEST = ROOT / 'results/monitor_cost_v3'
ORDER = ('a1', 'b1', 'b2', 'a2')


def plans(layout):
    result = []
    for label in ORDER:
        p = benchmark_plan(f'moncost3_{layout}_{label}', layout, 2026102316,
                           label[0].upper(), 'monitor',
                           'results/v2_semantic/linear_candidate_model.json', 'semantic_only')
        p['intervals'] = [dict(id='healthy', start=1, end=60, record_start=13,
                               target_ranks=[], delay_ms=0, segment='healthy')]
        result.append(p)
    return result


def score(layout):
    runs, losses, identities, checks = [], [], [], {}
    for p in plans(layout):
        directory = ROOT / 'results/data16_sessions' / p['session_id']
        rows = [json.loads(line) for line in (directory / 'host2/train.jsonl').read_text().splitlines()]
        manifests = [json.loads((directory / host / f'node{node}_manifest.json').read_text())
                     for node, host in enumerate(('host2', 'host1'))]
        checks[p['session_id'] + ':complete'] = all(m['status'] == 'completed' for m in manifests)
        checks[p['session_id'] + ':sequence'] = [r['iteration'] for r in rows] == list(range(1, 61))
        checks[p['session_id'] + ':no_profile'] = all(r['profile_epoch'] == 0 for r in rows)
        events = directory / 'host2/events.jsonl'
        ev = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        checks[p['session_id'] + ':no_trigger'] = not any(r['event'] in ('detection', 'profile_schedule', 'localization') for r in ev)
        keys = ('source_sha256', 'model_sha256', 'environment')
        checks[p['session_id'] + ':node_identity'] = all(manifests[0][k] == manifests[1][k] for k in keys)
        identities.append({k: manifests[0][k] for k in keys})
        losses.append([r['loss'] for r in rows])
        measured = rows[12:]
        runs.append(dict(session=p['session_id'], mode=p['benchmark_mode'],
                         mean_ms=statistics.mean(r['wall_iteration_ms'] for r in measured),
                         sum_ms=sum(r['wall_iteration_ms'] for r in measured),
                         all_measured_steps=48))
        if p['online']:
            import struct
            header = struct.unpack('<8s6Q', (directory / 'host2/monitor.bin').read_bytes()[:56])
            checks[p['session_id'] + ':consumed'] = header[1] == header[2] == 60 and header[3] == 0
            checks[p['session_id'] + ':sampling_off'] = all(int.from_bytes((directory / host / 'state.bin').read_bytes(), 'little') == 0 for host in ('host2', 'host1'))
    checks['identity_across_runs'] = all(x == identities[0] for x in identities)
    max_loss_delta = max(abs(x-y) for row in losses[1:] for x,y in zip(row,losses[0]))
    checks['numerics_match'] = max_loss_delta < 1e-4
    a = sum(r['sum_ms'] for r in runs if r['mode'] == 'A')
    b = sum(r['sum_ms'] for r in runs if r['mode'] == 'B')
    pairs = [100*(runs[1]['sum_ms']/runs[0]['sum_ms']-1),
             100*(runs[2]['sum_ms']/runs[3]['sum_ms']-1)]
    report = dict(layout=layout, protocol='same-seed/SGD full wall ABBA, all48 post-warmup steps, no injected/hook workload',
                  runs=runs, checks=checks, integrity_passed=all(checks.values()),
                  max_loss_delta=max_loss_delta, aggregate_overhead_pct=100*(b/a-1),
                  adjacent_pair_overhead_pct=pairs, paired_range_pct=[min(pairs),max(pairs)],
                  confidence_bound_available=False,
                  limitations=['Two adjacent pairs describe drift, not a statistical confidence interval.',
                               'A uses no FLASH probe, evidence collector or online controller; neither mode installs fault hooks.',
                               'Negative differences are not acceleration; no per-window or long-term guarantee.'])
    (DEST / f'{layout}_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    all_plans = [p for layout in ('legacy','node_local_pp') for p in plans(layout)]
    if args.run:
        protocol=DEST / 'protocol.json'
        if protocol.exists():
            raise FileExistsError(protocol)
        protocol.write_text(json.dumps(dict(plans=all_plans,
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'flash_v1/train_pipeline.py',Path(__file__)]}),indent=2)+'\n')
        for p in all_plans:
            run(p, timeout=600)
            print(json.dumps(dict(session=p['session_id'],completed=True)),flush=True)
    for layout in ('legacy','node_local_pp'):
        r=score(layout)
        print(json.dumps({k:r[k] for k in ('layout','integrity_passed','aggregate_overhead_pct','adjacent_pair_overhead_pct','max_loss_delta')}),flush=True)


if __name__ == '__main__':
    main()
