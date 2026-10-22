"""Disclosed paper-consistent offline baseline, independent of analysis.localize.

No healthy reference, fitted parameters, mandatory tag count, or pipeline veto.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics as st


AMBIGUITIES = [
    'Paper pp30-31/34 differs on the 20% rank gate; use (mean-min)/mean >= 0.2.',
    'Equal tag/kind/count/dtype comparison is a disclosed comparability adaptation.',
    'Keep supplied stage-aware tags verbatim; do not infer missing PP stages.',
    'Multiple shapes/communicators of one tag/name/member group contribute only '
    'the maximum normalized rank score; conflicting equal maxima abstain for that '
    'group only. This fusion and exact-tie abstention are unspecified by the paper.',
    'No-evidence means inconclusive, not a certified healthy diagnosis; Group '
    'Event timings may be shared by operations and are not independent samples.',
]


def _cv(values):
    return st.pstdev(values) / st.mean(values) if len(values) > 1 else 0.0


def localize(rows, cv_threshold=.1, rank_threshold=.2):
    """Accept topology-labeled rows; maximize clique latency, minimize rank latency.

    Communicator means precede clique max; rank means pool its observations as
    described on paper p34. Scores are dimensionless (mean-min)/mean. The caller
    must preserve the actual PP stage in tag; this function never pools tags.
    """
    if not (math.isfinite(cv_threshold) and cv_threshold >= 0 and
            math.isfinite(rank_threshold) and 0 <= rank_threshold <= 1):
        raise ValueError('invalid CV/rank threshold')
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    memberships, names = {}, {}
    for row in rows:
        tag = str(row['tag'])
        match = re.fullmatch(r'(.*):count([^:]+):dtype([^:]+)', tag)
        count, dtype = row.get('count'), row.get('dtype')
        if match:
            tag, count, dtype = match[1], count if count is not None else match[2], dtype if dtype is not None else match[3]
        kind = row.get('kind', row.get('operation'))
        if kind is None or count is None or dtype is None:
            raise ValueError('paper baseline requires explicit kind/operation, count and dtype')
        rank = row['rank']
        members = tuple(sorted(row['group_members']))
        if type(rank) is not int or rank not in members or len(set(members)) != len(members):
            raise ValueError('invalid labeled group membership')
        value = float(row['duration_us'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('duration_us must be positive finite')
        key = (tag, str(kind), str(count), str(dtype))
        clique = str(row['clique'])
        identity = (key, clique)
        name = str(row.get('group_name', tag))
        if identity in memberships and (memberships[identity] != members or names[identity] != name):
            raise ValueError('inconsistent labeled clique')
        memberships[identity], names[identity] = members, name
        comm = str(row.get('comm', row.get('comm_addr', rank)))
        groups[key][clique][rank, comm].append(value)
    evidence, skipped = [], []
    for key, cliques in sorted(groups.items()):
        tag, kind, count, dtype = key
        detail = dict(tag=tag, kind=kind, count=count, dtype=dtype)
        valid = {}
        for clique, data in sorted(cliques.items()):
            members = memberships[key, clique]
            if len(members) < 2 or set(members) != {rank for rank, _ in data}:
                skipped.append(dict(detail, clique=clique, reason='singleton or incomplete comparable shape'))
            else:
                valid[clique] = data
        if not valid:
            continue
        maxima = {clique: max(st.mean(xs) for xs in data.values()) for clique, data in valid.items()}
        group_cv = _cv(list(maxima.values()))
        if len(valid) > 1 and group_cv < cv_threshold:
            skipped.append(dict(detail, reason='clique CV below threshold', cv=group_cv))
            continue
        choices = [clique for clique, value in maxima.items() if value == max(maxima.values())]
        if len(choices) != 1:
            skipped.append(dict(detail, reason='maximum clique tied', cliques=choices))
            continue
        clique = choices[0]
        per_rank = defaultdict(list)
        for (rank, _), xs in valid[clique].items():
            per_rank[rank].extend(xs)
        averages = {rank: st.mean(xs) for rank, xs in sorted(per_rank.items())}
        mean = st.mean(averages.values())
        rank_cv, minimum = _cv(list(averages.values())), min(averages.values())
        choices = [rank for rank, value in averages.items() if value == minimum]
        score = (mean - minimum) / mean
        if rank_cv < cv_threshold or score < rank_threshold or len(choices) != 1:
            skipped.append(dict(detail, clique=clique, reason='insufficient rank separation or tied minimum',
                                rank_cv=rank_cv, score=score, rank_means_us=averages))
            continue
        evidence.append(dict(detail, clique=clique, members=list(memberships[key, clique]),
                             group_name=names[key, clique], rank=choices[0], score=score,
                             rank_cv=rank_cv, rank_means_us=averages, clique_cv=group_cv,
                             clique_maxima_us=maxima, group_screened=len(valid) > 1))
    real_groups = defaultdict(list)
    for item in evidence:
        real_groups[item['tag'], item['group_name'], tuple(item['members'])].append(item)
    scores, tags, contributions = defaultdict(float), defaultdict(set), []
    for (tag, name, members), items in sorted(real_groups.items()):
        best = max(item['score'] for item in items)
        winners = {item['rank'] for item in items if item['score'] == best}
        winner = next(iter(winners)) if len(winners) == 1 else None
        contributions.append(dict(tag=tag, group_name=name, members=list(members),
                                  rank=winner, score=best if winner is not None else 0,
                                  shape_candidates=sorted({item['rank'] for item in items})))
        if winner is not None:
            scores[winner] += best
            tags[winner].add(tag)
    ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
    tied = len(ordered) > 1 and math.isclose(scores[ordered[0]], scores[ordered[1]], rel_tol=1e-12)
    winner = ordered[0] if ordered and not tied else None
    return dict(method='paper_consistent_disclosed_baseline', rank=winner,
                status='identified' if winner is not None else 'inconclusive',
                reason='unique maximum normalized score' if winner is not None else 'no evidence or tied rank scores',
                thresholds=dict(cv=cv_threshold, rank=rank_threshold),
                candidates=[dict(rank=rank, score=scores[rank], tags=sorted(tags[rank])) for rank in ordered],
                groups=contributions, evidence=evidence, skipped=skipped, paper_ambiguities=AMBIGUITIES)


def evaluate_dataset(dataset, split='validation'):
    """Match offline.py episode layout; truth is accessed after all predictions."""
    from flash_v1.offline import _catalog_runs, _label_from_manifest, _profile_rows, _read_json
    from flash_v1.topology import label_cliques
    catalog = _catalog_runs(Path(dataset).resolve(), load_labels_for=set())
    selected = [run for run in catalog if run['role'] == split]
    if not selected:
        raise ValueError(f'dataset has no {split} episodes')
    predictions = []
    for run in selected:
        folder = run['run_dir']
        rows = _profile_rows(_read_json(folder / 'profile.json'))
        accepted, diagnostics = label_cliques(rows, _read_json(folder / 'topology.json'))
        result = localize(accepted)
        predictions.append(dict(run_id=run['run_id'], result=result, topology_diagnostics=diagnostics))
    # Scoring is separate from feature loading and localization.
    for run, prediction in zip(selected, predictions):
        prediction['fault_rank'] = _label_from_manifest(_read_json(run['manifest_path']))
    faults = [run for run in predictions if run['fault_rank'] is not None]
    healthy = [run for run in predictions if run['fault_rank'] is None]
    correct = sum(run['result']['rank'] == run['fault_rank'] for run in faults)
    fp = sum(run['result']['rank'] is not None for run in healthy)
    rejected = sum(run['result']['rank'] is None for run in predictions)
    return dict(method='paper_consistent_disclosed_baseline', split=split,
                fault_localization_accuracy=correct / len(faults) if faults else None,
                correct_fault_runs=correct, fault_runs=len(faults),
                inconclusive_runs=rejected, episode_runs=len(predictions),
                rejection_rate=rejected / len(predictions),
                healthy_false_positives=fp, healthy_runs=len(healthy),
                healthy_false_positive_rate=fp / len(healthy) if healthy else None,
                healthy_interpretation='No rank prediction is no alarm/inconclusive, not proven healthy.',
                runs=predictions, paper_ambiguities=AMBIGUITIES)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'validation', 'blind_test'], default='validation')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_dataset(args.dataset, args.split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in report.items() if key not in {'runs', 'paper_ambiguities'}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
