"""Opt-in v2 evidence plans; GPU launch requires --run and a fresh session."""
import argparse
import json
from pathlib import Path
import random

from flash_v1.scripts.data16 import ROOT, run

LAYOUTS = ('legacy', 'node_local_pp')
MODES = ('attention_fwd_sleep', 'experts_bwd_sleep', 'attention_bwd_sleep',
         'attention_bwd_compute', 'whole_compute')


def interval(name, start, length, **fields):
    row = dict(id=name, start=start, end=start + length - 1, record_start=start,
               target_ranks=[], delay_ms=0., fault_mode=None, segment='healthy')
    row.update(fields)
    return row


def plan(session, layout, role, seed, kind='healthy', mode_ranks=None, *,
         monitor_confirm_recent_three=True):
    if type(monitor_confirm_recent_three) is not bool:
        raise ValueError('monitor_confirm_recent_three must be a bool')
    if layout not in LAYOUTS or role not in ('train', 'validation', 'development'):
        raise ValueError('invalid layout/role')
    rows = [interval('healthy_opening', 1, 12, record_start=13, warmup=True)]
    if kind == 'healthy':
        rows += [interval(f'healthy_window{i}', 13 + 8*i, 8) for i in range(6)]
    elif kind == 'smoke':
        rng = random.Random(seed)
        ranks = rng.sample(range(16), len(MODES)) if mode_ranks is None else mode_ranks
        if len(ranks) != len(MODES) or len(set(ranks)) != len(ranks):
            raise ValueError('one distinct rank per mode required')
        start = 13
        formal = json.loads((ROOT / 'results/whole32/frozen' / f'formal_{layout}.json').read_text())
        ratios = {r['target_ranks'][0]: r['slowdown_extra_ratio'] for r in formal['intervals'] if r.get('target_ranks')}
        for i, (mode, rank) in enumerate(zip(MODES, ranks)):
            rows.append(interval(f'healthy_before_{i}', start, 8)); start += 8
            rows.append(interval(f'dev_{i}_{mode}', start, 12, record_start=start + 4,
                                 target_ranks=[rank], fault_mode=mode, segment='fault',
                                 delay_ms=0 if mode == 'whole_compute' else 450.,
                                 slowdown_extra_ratio=ratios[rank] if mode == 'whole_compute' else 0.))
            start += 12
    else:
        raise ValueError('unknown collection kind')
    result = dict(schema_version=1, protocol='v2_evidence_development', session_id=session,
                  parallel_layout=layout, role=role, seed=seed, seq_len=16,
                  monitor_confirm_recent_three=monitor_confirm_recent_three,
                  online=False, probe_enabled=True, profile_enabled=True, evidence_enabled=True,
                  fault_kind='mixed_compute_simulation',
                  inference_exclusions=['target_ranks', 'fault_mode', 'delay_ms', 'slowdown_extra_ratio'],
                  intervals=rows)
    if kind == 'smoke':
        result['whole_compute_reference_session'] = ('whole32_pilot_legacy_01' if layout == 'legacy'
                                                     else 'whole32_pilot_node_local_pp_02')
    return result


def online_validation_plan(session, layout, seed, model, *, monitor_confirm_recent_three=True):
    """Two previously developed fault modes through the unchanged online loop."""
    result = plan(session, layout, 'development', seed, 'healthy',
                  monitor_confirm_recent_three=monitor_confirm_recent_three)
    result.update(protocol='v2_online_development_validation', online=True,
                  profile_enabled=False, evidence_model=model, max_profiles=3,
                  intervals=[interval('healthy_opening',1,45),
                    interval('fwd_fault',46,25,target_ranks=[14],fault_mode='attention_fwd_sleep',delay_ms=600.,segment='fault'),
                    interval('healthy_recovery',71,35),
                    interval('expert_fault',106,25,target_ranks=[3],fault_mode='experts_bwd_sleep',delay_ms=600.,segment='fault'),
                    interval('healthy_final',131,15)])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True)
    p.add_argument('--layout', choices=LAYOUTS, required=True)
    p.add_argument('--role', choices=('train', 'validation', 'development'), required=True)
    p.add_argument('--kind', choices=('healthy', 'smoke'), default='healthy')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--run', action='store_true')
    p.add_argument('--monitor-confirm-recent-three', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    value = plan(args.session, args.layout, args.role, args.seed, args.kind,
                 monitor_confirm_recent_three=args.monitor_confirm_recent_three)
    output = ROOT / 'results/v2_collection/plans' / f'{args.session}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(plan=str(output), steps=value['intervals'][-1]['end'], launched=args.run)))
    if args.run:
        run(value, timeout=args.timeout)


if __name__ == '__main__':
    main()
