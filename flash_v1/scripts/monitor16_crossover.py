"""Controlled same-process MONITOR crossover; training starts only with --run."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


ROOT = Path(__file__).resolve().parents[2]
ORDER = ('A', 'B', 'B', 'A', 'B', 'A', 'A', 'B')


def benchmark_blocks():
    return [dict(block=index + 1, mode=mode, start=13 + index * 32, end=44 + index * 32)
            for index, mode in enumerate(ORDER)]


def benchmark_step(iteration):
    if not 1 <= iteration <= 268:
        raise ValueError('crossover iteration outside 1..268')
    if iteration <= 12:
        return dict(monitor_active=False, benchmark_block=0, benchmark_position=iteration)
    index, position = divmod(iteration - 13, 32)
    return dict(monitor_active=ORDER[index] == 'B', benchmark_block=index + 1,
                benchmark_position=position + 1)


def make_plan(session, seed):
    blocks = benchmark_blocks()
    intervals = [dict(id='warmup', start=1, end=12, record_start=1,
                      target_ranks=[], delay_ms=0, warmup=True)]
    intervals += [dict(id=f"block{block['block']}_{block['mode']}", start=block['start'],
                       end=block['end'], record_start=block['start'], target_ranks=[], delay_ms=0)
                  for block in blocks]
    return dict(session_id=session, role='monitor_controlled_benchmark', online=True,
                monitor_benchmark=True, benchmark_blocks=blocks, probe_enabled=True,
                profile_enabled=False, max_profiles=3, seq_len=16, seed=seed,
                fault_kind='none_controlled_measurement', intervals=intervals)


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def percentile(values, fraction):
    values = sorted(values)
    point = (len(values) - 1) * fraction
    return values[math.floor(point)] + (values[math.ceil(point)] - values[math.floor(point)]) * (point - math.floor(point))


def timing_statistics(rows):
    by_block = {block: [row for row in rows if row['benchmark_block'] == block] for block in range(1, 9)}
    for block, values in by_block.items():
        if [row['benchmark_position'] for row in values] != list(range(1, 33)):
            raise ValueError(f'block {block} is not a complete 32-position sequence')
        if any(not math.isfinite(row['wall_iteration_ms']) or row['wall_iteration_ms'] <= 0 for row in values):
            raise ValueError('wall durations must be finite positive')
    block_reports = [dict(block=index + 1, mode=mode,
                          sum_ms=sum(row['wall_iteration_ms'] for row in by_block[index + 1]))
                     for index, mode in enumerate(ORDER)]
    positions = []
    for position in range(32):
        a = [by_block[index + 1][position]['wall_iteration_ms'] for index, mode in enumerate(ORDER) if mode == 'A']
        b = [by_block[index + 1][position]['wall_iteration_ms'] for index, mode in enumerate(ORDER) if mode == 'B']
        losses = [float(by_block[index + 1][position]['loss']) for index in range(8)]
        positions.append(dict(position=position + 1, baseline_sum_ms=sum(a), online_sum_ms=sum(b),
                              ratio=sum(b) / sum(a), loss_spread=max(losses) - min(losses),
                              loss_finite=all(math.isfinite(value) for value in losses)))
    pairs = []
    for first in range(0, 8, 2):
        pair = block_reports[first:first + 2]
        a = next(row for row in pair if row['mode'] == 'A')
        b = next(row for row in pair if row['mode'] == 'B')
        pairs.append(dict(blocks=[first + 1, first + 2], order=ORDER[first] + ORDER[first + 1],
                          baseline_block=a['block'], online_block=b['block'],
                          overhead_pct=100 * (b['sum_ms'] / a['sum_ms'] - 1)))
    total_a = sum(row['baseline_sum_ms'] for row in positions)
    total_b = sum(row['online_sum_ms'] for row in positions)
    ratios = [row['ratio'] for row in positions]
    return dict(baseline_sum_ms=total_a, online_sum_ms=total_b,
                aggregate_overhead_pct=100 * (total_b / total_a - 1),
                position_ratio_median=statistics.median(ratios), position_ratio_p95=percentile(ratios, .95),
                position_ratio_range=[min(ratios), max(ratios)], positions=positions,
                blocks=block_reports, adjacent_pairs=pairs,
                neighboring_block_ratios=[dict(blocks=[index, index + 1],
                                               ratio=block_reports[index]['sum_ms'] / block_reports[index - 1]['sum_ms'])
                                          for index in range(1, 8)],
                max_position_loss_spread=max(row['loss_spread'] for row in positions),
                all_losses_finite=all(row['loss_finite'] for row in positions))


def score(local):
    local = Path(local)
    required = [local / 'plan.json', local / 'host2/train.jsonl', local / 'host2/run.json']
    required += [local / host / name for node, host in enumerate(('host2', 'host1'))
                 for name in (f'node{node}_manifest.json', 'drain_status.json', 'state.bin', 'raw_profile.jsonl.gz')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='missing', missing=missing)
    plan, run = read_json(local / 'plan.json'), read_json(local / 'host2/run.json')
    rows = jsonl(local / 'host2/train.jsonl')
    events = jsonl(local / 'host2/events.jsonl')
    checks = dict(protocol=plan.get('monitor_benchmark') is True and plan.get('benchmark_blocks') == benchmark_blocks(),
                  exact_iteration_sequence=[row['iteration'] for row in rows] == list(range(1, 269)),
                  controlled_lr_zero=run.get('lr') == 0 and run.get('monitor_benchmark') is True,
                  healthy_plan=not any(row.get('target_ranks') or row.get('delay_ms', 0) for row in plan['intervals']),
                  no_profile_steps=all(row.get('profile_epoch') == 0 for row in rows),
                  no_detection_or_profile_events=not any(row['event'] in ('detection', 'profile_schedule', 'localization') for row in events))
    checks['recorded_modes_and_positions'] = all(
        all(row.get(key) == value for key, value in benchmark_step(row['iteration']).items()) for row in rows)
    manifests, drains = [], {}
    for node, host in enumerate(('host2', 'host1')):
        manifest = read_json(local / host / f'node{node}_manifest.json')
        manifests.append(manifest)
        drains[host] = read_json(local / host / 'drain_status.json')
        checks[f'{host}:completed'] = manifest.get('status') == 'completed'
        checks[f'{host}:zero_ring_drops'] = drains[host].get('status') == 'complete' and drains[host].get('dropped') == []
        checks[f'{host}:sampling_disabled'] = int.from_bytes((local / host / 'state.bin').read_bytes(), 'little') == 0
        with gzip.open(local / host / 'raw_profile.jsonl.gz', 'rt') as stream:
            checks[f'{host}:no_timed_profile_rows'] = not any(line.strip() for line in stream)
    for key in ('configuration', 'source_sha256', 'model_sha256', 'probe_sha256', 'snapshot_sha256', 'environment'):
        checks[f'aligned:{key}'] = key in manifests[0] and manifests[0].get(key) == manifests[1].get(key)
    checks['controller_benchmark_configuration'] = manifests[0]['configuration'].get('monitor_benchmark_blocks') == benchmark_blocks()
    channel = local / 'host2/monitor.bin'
    channel_summary = None
    if channel.is_file():
        header = struct.unpack('<8s6Q', channel.read_bytes()[:56])
        channel_summary = dict(version=header[0].decode(), published=header[1], consumed=header[2], request_epoch=header[3])
    checks['all_measurement_rows_consumed'] = (channel_summary is not None and channel_summary['version'] == 'FLASHM01'
                                              and channel_summary['published'] == channel_summary['consumed'] == 268
                                              and channel_summary['request_epoch'] == 0)
    timings = None
    if checks['exact_iteration_sequence'] and checks['recorded_modes_and_positions']:
        timings = timing_statistics(rows[12:])
    checks['position_losses_match'] = (timings is not None and timings['all_losses_finite']
                                        and timings['max_position_loss_spread'] < 1e-4)
    report = dict(session=plan['session_id'], status='complete' if all(checks.values()) else 'validation_failed',
                  checks=checks, failed_checks=[key for key, valid in checks.items() if not valid],
                  protocol=dict(warmup=12, blocks=benchmark_blocks(), lr=0, repeated_token_positions=32),
                  timing=timings, drain_evidence=drains, channel=channel_summary,
                  metadata_sha256={str(path.relative_to(local)): hashlib.sha256(path.read_bytes()).hexdigest() for path in required
                                   if path.suffix in ('.json', '.jsonl')},
                  incremental_target_met=all(checks.values()) and timings['aggregate_overhead_pct'] < 1,
                  total_no_probe_target_certified=False,
                  limitations=['A shares the native MONITOR fast path and measurement channel; it is not a no-probe baseline.',
                               'B uses the production MONITOR metric/control path and per-step CPU BEAST; no PROFILE is measured.',
                               'Learning rate zero and repeated tokens control workload variation; production SGD detection accuracy requires separate fault validation.',
                               'Short serial blocks on shared GPUs retain temporal drift; all blocks and paired results are reported.',
                               'Asynchronous CPU monitoring work may spill slightly across block boundaries; no boundary positions are discarded.',
                               'Wall intervals include the previous step logging/publication tail, including at mode boundaries.'])
    (local / 'monitor_crossover_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='monopt_crossover_01')
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--run', action='store_true', help='Explicitly authorize launching the two-node controlled benchmark')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    local = ROOT / 'results/data16_sessions' / args.session
    if args.run:
        from flash_v1.scripts.data16 import run
        local = run(make_plan(args.session, args.seed), timeout=args.timeout)
    report = score(local)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
