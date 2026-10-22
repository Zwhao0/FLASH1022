"""Fixed, one-shot eight-rank PP validation; no retries or threshold tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(directory, rank, reference_sha, returncode=None, error=None):
    manifest = read_json(directory / 'manifest.json')
    events = read_lines(directory / 'events.jsonl')
    truth = [row for path in sorted(directory.glob('fault_rank*.jsonl')) for row in read_lines(path)]
    starts = [row['iteration'] for row in truth if row.get('type') == 'start']
    ends = [row['iteration'] for row in truth if row.get('type') == 'end']
    end_wall = next((row.get('walltime') for row in truth if row.get('type') == 'end'), None)
    start_wall = next((row.get('walltime') for row in truth if row.get('type') == 'start'), None)
    first = min(starts) if starts else 46
    detections = [row for row in events if row.get('event') == 'detection']
    localizations = {row['epoch']: row for row in events if row.get('event') == 'localization'}
    observations = []
    for epoch, row in enumerate(detections, 1):
        localization = localizations.get(epoch, {})
        result = localization.get('result', {})
        during = rank is None or (start_wall is not None and end_wall is not None
                                 and start_wall <= localization.get('wall_time', 0) < end_wall)
        observations.append(dict(epoch=epoch, detection_step=row.get('iteration'), cp=row.get('cp'),
                                 predicted_rank=result.get('rank'), status=result.get('status', 'missing'),
                                 cp_matches_start=row.get('cp') == first, localization_during_fault=during,
                                 pre_fault=rank is not None and row.get('iteration', 0) < first))
    eligible = [row for row in observations if not row['pre_fault']]
    wrong = rank is not None and any(row['predicted_rank'] is not None and row['predicted_rank'] != rank for row in eligible)
    inconclusive = any(row['predicted_rank'] is None or not row['localization_during_fault'] for row in eligible)
    no_detection = not eligible
    problems = [error, manifest.get('error')]
    if manifest.get('reference_profile_sha256') != reference_sha:
        problems.append('reference SHA mismatch or missing')
    if rank is not None and (not starts or not ends or any(row.get('rank') != rank for row in truth)):
        problems.append('missing or mismatched injected fault truth')
    failed = manifest.get('status') != 'completed' or returncode not in (None, 0) or any(problems)
    status = ('failed' if failed else 'wrong' if wrong else 'false_positive' if (rank is None and detections) or any(row['pre_fault'] for row in observations)
              else 'no_detection' if rank is not None and no_detection else 'inconclusive' if inconclusive else 'correct')
    return dict(name=directory.name, truth_rank=rank, status=status, manifest_status=manifest.get('status'),
                error='; '.join(str(item) for item in problems if item) or None, returncode=returncode,
                wrong=wrong, inconclusive=inconclusive, no_detection=no_detection, observations=observations,
                pre_fault_triggers=sum(row['pre_fault'] for row in observations), truth=truth,
                reference_profile_sha256=manifest.get('reference_profile_sha256'),
                source_sha256=manifest.get('source_sha256'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-profile', type=Path, default=Path('results/stage3_pp_v2_healthy_profile/profile_1.json'))
    parser.add_argument('--redis-port', type=int, default=16430)
    parser.add_argument('--resume', action='store_true', help='Skip completed runs only; never overwrite any run directory')
    args = parser.parse_args()
    root, reference = args.output.resolve(), args.reference_profile.resolve()
    reference_sha = digest(reference)
    names = [('healthy', None)] + [(f'rank{rank}', rank) for rank in range(8)]
    if root.exists() and not args.resume:
        parser.error('output exists; use a new directory or explicit --resume')
    if reference.is_relative_to(root):
        parser.error('reference must be independent of this matrix')
    for name, _ in names:
        directory = root / name
        if directory.exists() and (not args.resume or read_json(directory / 'manifest.json').get('status') != 'completed'):
            parser.error(f'refusing to overwrite incomplete/existing run: {directory}')
    root.mkdir(parents=True, exist_ok=True)
    matrix = read_json(root / 'matrix.json') if args.resume else {}
    if matrix and matrix.get('reference_profile_sha256') != reference_sha:
        parser.error('resume reference SHA differs from saved matrix')
    matrix.update(reference_profile=str(reference), reference_profile_sha256=reference_sha,
                  order=[name for name, _ in names])
    matrix.setdefault('runs', [])
    environment = os.environ.copy()
    environment.update(NCCL_MAX_NCHANNELS='4', NCCL_BUFFSIZE='1048576')
    frozen = read_json(root / 'frozen_sources.json')
    for name, rank in names:
        directory, error, returncode = root / name, None, None
        skipped = directory.exists()
        if not skipped:
            command = [sys.executable, '-m', 'flash_v1.run', '--mode', 'monitor' if rank is None else 'fault',
                       '--output', str(directory), '--gpus', '0,1,2,3,4,5,6,7', '--gradient-backend', 'pipeline',
                       '--delay-placement', 'attention-experts', '--microbatches', '2', '--steps', '110',
                       '--inject-at', '45', '--fault-steps', '60', '--fault-kind', 'gpu-delay', '--delay-ms', '125',
                       '--fault-rank', str(rank or 0), '--redis-port', str(args.redis_port),
                       '--reference-profile', str(reference)]
            temporary_log = root / f'{name}_runner.log'
            try:
                if digest(reference) != reference_sha:
                    raise RuntimeError('reference changed during matrix')
                with temporary_log.open('x') as log:
                    returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                error = repr(exc)
            finally:
                directory.mkdir(exist_ok=True)
                if temporary_log.exists():
                    temporary_log.replace(directory / 'runner.log')
        try:
            row = summarize(directory, rank, reference_sha, returncode, error)
        except Exception as exc:
            row = dict(name=name, truth_rank=rank, status='failed', error=repr(exc))
        if frozen and any((row.get('source_sha256') or {}).get(path) != sha for path, sha in frozen.items()):
            row.update(status='failed', error='source SHA differs from frozen_sources.json')
        row['skipped'] = skipped
        previous = next((item for item in matrix['runs'] if item['name'] == name), None)
        if previous and skipped:
            row = previous
        matrix['runs'] = [item for item in matrix['runs'] if item['name'] != name] + [row]
        temporary = root / 'matrix.json.tmp'
        temporary.write_text(json.dumps(matrix, indent=2) + '\n')
        temporary.replace(root / 'matrix.json')
        print(json.dumps({key: value for key, value in row.items() if key not in ('truth', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
