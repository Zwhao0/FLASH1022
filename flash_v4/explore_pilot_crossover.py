"""Compact A/B crossover summary using the existing compute-envelope matcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flash_v4.explore_pilot_compute import analyze, _output_is_within_input


def _pair_summary(result: dict[str, Any]) -> dict[str, Any]:
    ranks = result['ranks']
    ratios = [row['median_descriptor_time_ratio'] for row in ranks
              if row['median_descriptor_time_ratio'] is not None]
    coverage = [row['paired_sample_coverage']['conservative_min_side_percent']
                for row in ranks
                if row['paired_sample_coverage']['conservative_min_side_percent'] is not None]
    return {
        'rank_median_effect_ratio_range': (
            [round(min(ratios), 6), round(max(ratios), 6)] if ratios else None),
        'ranks_with_comparable_data': len(ratios),
        'common_descriptor_count_range': (
            [min(row['common_descriptor_count'] for row in ranks),
             max(row['common_descriptor_count'] for row in ranks)] if ranks else None),
        'conservative_coverage_percent_range': (
            [round(min(coverage), 3), round(max(coverage), 3)] if coverage else None),
        'rank_sample_counts': [
            {
                'rank': row['rank'],
                'left_valid': row['valid_health_sample_count'],
                'right_valid': row['valid_condition_sample_count'],
                'common_descriptors': row['common_descriptor_count'],
                'conservative_coverage_percent': row['paired_sample_coverage'][
                    'conservative_min_side_percent'],
                'median_ratio': row['median_descriptor_time_ratio'],
            }
            for row in ranks
        ],
    }


def analyze_crossover(a1: Path, a2: Path, b1: Path, b2: Path) -> dict[str, Any]:
    pairs = {
        'A2_over_A1_self_drift': (a1, a2),
        'B1_over_A1': (a1, b1),
        'B2_over_A1': (a1, b2),
        'B1_over_A2': (a2, b1),
        'B2_over_A2': (a2, b2),
    }
    comparisons = {}
    for name, (left, right) in pairs.items():
        result = analyze(left, right)
        comparisons[name] = {
            'left': 'A1' if left == a1 else 'A2',
            'right': name.split('_over_', 1)[0],
            'valid_samples': {
                'left': result['input_counts']['health_reference']['valid_target_records'],
                'right': result['input_counts']['condition_comparison']['valid_target_records'],
            },
            **_pair_summary(result),
        }
    return {
        'analysis': 'paired_run_crossover_compute_envelope_summary',
        'status': 'descriptive_only_no_pass_or_location_claim',
        'matching_fields': ['rank', 'kind', 'op', 'dtype', 'shape'],
        'sequence_used_as_operator_identity': False,
        'minimum_samples_per_descriptor': 5,
        'coverage_semantics': 'matched-descriptor sample coverage only; window completion is not assessed',
        'interpretation': 'same-descriptor distributions; ratios are B/A run contrasts, not causal proof',
        'inputs': {'A1': str(a1), 'A2': str(a2), 'B1': str(b1), 'B2': str(b2)},
        'comparisons': comparisons,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('a1', 'a2', 'b1', 'b2'):
        parser.add_argument(f'--{label}', required=True, type=Path,
                            help=f'{label.upper()} run root with node*/observations/rank_*.jsonl')
    parser.add_argument('--output', required=True, type=Path,
                        help='new JSON report path; existing files are never overwritten')
    args = parser.parse_args(argv)
    inputs = (args.a1, args.a2, args.b1, args.b2)
    if len({path.resolve() for path in inputs}) != 4:
        parser.error('A1, A2, B1 and B2 must identify four different runs')
    if any(_output_is_within_input(args.output, root) for root in inputs):
        parser.error('--output must be outside all input directories')
    if args.output.exists():
        parser.error('--output already exists; refusing to overwrite it')
    try:
        result = analyze_crossover(*inputs)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    except FileExistsError:
        parser.error('--output was created concurrently; refusing to overwrite it')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
