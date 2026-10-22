"""Replay saved PP observations; preserve live and original algorithm results."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

from flash_v1.analysis import _profile_key, localize


def audit(directory, reference=None, legacy_archive=None):
    rows = json.loads((directory / 'profile_1.json').read_text())
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    recorded = next(e['result'] for e in events if e['event'] == 'localization')
    topology = json.loads((directory / 'topology.json').read_text())
    baseline = json.loads(reference.read_text()) if reference else None
    replay = localize(rows, topology=topology, baseline_rows=baseline)
    coverage = defaultdict(Counter)
    for row in rows:
        coverage[str(_profile_key(row))][row['rank']] += 1
    result = dict(source=str(directory), samples=len(rows), recorded=recorded, replay=replay,
                  recorded_rank_matches=recorded['rank'] == replay['rank'],
                  coverage={key: dict(value) for key, value in coverage.items()},
                  reference=str(reference) if reference else None,
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None)
    if legacy_archive:
        # Only the reviewed original snapshot from this reproduction is used.
        # Execute it in an isolated namespace; never extract or overwrite sources.
        with tarfile.open(legacy_archive) as archive:
            source = archive.extractfile('flash_v1/analysis.py').read()
        namespace = {}
        exec(compile(source, str(legacy_archive) + ':flash_v1/analysis.py', 'exec'), namespace)
        legacy_rows = []
        for row in rows:
            family, count, dtype = _profile_key(row)
            tag = f'{family}:count{count}:dtype{dtype}'
            legacy_rows.append(dict(row, tag=tag))
        result['original_algorithm'] = namespace['localize'](legacy_rows)
        result['original_source_sha256'] = hashlib.sha256(source).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-profile', type=Path)
    parser.add_argument('--legacy-archive', type=Path)
    args = parser.parse_args()
    result = audit(args.directory, args.reference_profile, args.legacy_archive)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(samples=result['samples'], recorded_rank=result['recorded']['rank'],
                          replay_rank=result['replay']['rank'],
                          original_rank=result.get('original_algorithm', {}).get('rank'),
                          output=str(args.output))))


if __name__ == '__main__':
    main()
