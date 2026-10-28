"""Post-evaluation coverage audit only; excluded from candidate prediction.

Checks whether unmodeled samples were collected but excluded for too few health
shapes. Reads only H1 development references and six already-scored silent cases.
Exact-shape availability is a data diagnostic, not a new localization algorithm.
"""
from collections import Counter
import json
from pathlib import Path

from flash_v4.health_centered_fusion import fingerprint, write_new
from flash_v4.offline_localizer import _kernel_family_and_features


def kernels(path):
    result = []
    for line in path.open():
        row = json.loads(line)
        if row.get('kind') == 'kernel':
            key, sample, reason = _kernel_family_and_features(row)
            if reason is None:
                result.append((key, sample))
    return result


def run():
    root = Path('results/v4/compat/v4_delay_20261027_r11')
    evaluation_path = Path('results/v4/evaluation/r11_development_v2/alarm_evaluation.json')
    evaluation = json.loads(evaluation_path.read_text())
    healthy, sources = {}, [fingerprint(evaluation_path), fingerprint(Path(__file__))]
    for topology, session in (('legacy', 'dl_h1'), ('node_local_pp', 'dp_h1')):
        paths = sorted((root / session).glob('node*/observations/rank_*.jsonl'))
        if len(paths) != 16:
            raise ValueError('expected development H1 reference for all ranks')
        rows = [item for path in paths for item in kernels(path)]
        healthy[topology] = {key for key, _ in rows}
        sources.extend(fingerprint(p) for p in paths)
    cases = []
    for case in evaluation['development_fault_cases']:
        if not case['evaluable'] or case['alarm_rank'] is not None:
            continue
        session = case['session_id'].lower()
        if not (session.startswith('dl_') or session.startswith('dp_')):
            raise ValueError('audit is restricted to development sessions')
        paths = list((root / session).glob(f"node*/observations/rank_{case['target_rank']}.jsonl"))
        if len(paths) != 1:
            raise ValueError('one observation file expected')
        prediction_path = Path('results/v4/exploration/r11_development_v0') / (case['case_key'] + '.json')
        prediction = json.loads(prediction_path.read_text())
        models = {tuple(m['family']) for m in prediction['branches']['kernel']['models']}
        rows = kernels(paths[0])
        counts = Counter()
        for key, sample in rows:
            if key[0] in models:
                counts['modeled_family'] += 1
            else:
                counts['unmodeled_family'] += 1
                if key in healthy[case['topology']]:
                    counts['unmodeled_but_exact_shape_seen_in_H1'] += 1
        cases.append({'case_key': case['case_key'], 'target_rank': case['target_rank'],
                      'sampled_kernel_event_counts': dict(counts),
                      'health_fit_exclusions': prediction['branches']['kernel']['unfit_health_family_rows']})
        sources.extend((fingerprint(paths[0]), fingerprint(prediction_path)))
    output = evaluation_path.with_name('observation_coverage_audit.json')
    write_new(output, {
        'schema': 'v4-development-post-evaluation-coverage-audit-1',
        'scope': 'coverage_diagnostic_only_not_used_for_predictions_or_thresholds',
        'validation_read': False, 'sources': sources, 'cases': cases,
        'caveats': ['exact shape does not prove timing comparability or independence',
                    'sampled event coverage is not full kernel-time or framework coverage',
                    'remaining misses cannot be attributed causally from this audit alone'],
    })
    print(json.dumps({'cases': cases, 'output': str(output)}))


if __name__ == '__main__':
    run()
