"""One observation-only onset/continuation hint; frozen localizer is unchanged."""
from flash_v1 import localize_v2 as common, localize_v2_linear as linear


def candidate(rows, model, expected_ranks, schedule, result):
    if (model.get('model_type') != linear.MODEL_TYPE or result.get('status') != 'localized'
            or result.get('rank') is None or schedule['end'] - schedule['start'] != 7):
        return None
    from flash_v1.online_v2 import evidence_integrity_reason, ROLES
    if evidence_integrity_reason(rows, schedule, expected_ranks) is not None:
        return None
    scores, known = {}, {}
    for row in rows:
        if not common._usable(row):
            return None
        shape, dtype = row.get('shape'), row.get('dtype')
        if (not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(dtype, str) or not dtype.strip()):
            return None  # Unknown references may be skipped only for valid metadata.
        reference = model['baselines'].get(common.semantic_key(row)) or linear._interpolated(row, model)
        if reference is None:
            if row['rank'] == result['rank']:
                return None  # The target still requires comparable evidence.
            continue  # Match infer: unrelated unknown branches contribute no score.
        step, rank = row['iteration'], row['rank']
        z = (row['duration_us'] - reference['median_us']) / reference['scale_us']
        per_rank = scores.setdefault(step, {})
        per_rank[rank] = max(per_rank.get(rank, 0.), z)
        known.setdefault((step, rank), set()).add((row['slot'], row['phase']))
    rank, first, end = result['rank'], schedule['start'], schedule['end']
    if known.get((first, rank)) != ROLES or scores[first][rank] > model['min_score']:
        return None
    tail_scores, margins = [], []
    for step in range(end - 2, end + 1):
        ordered = sorted(scores[step], key=scores[step].get, reverse=True)
        if len(ordered) < 2 or ordered[0] != rank or known.get((step, rank)) != ROLES:
            return None
        value = scores[step][rank]
        margin = value - scores[step][ordered[1]]
        if value <= model['min_score'] or margin <= model['margin']:
            return None
        tail_scores.append(value)
        margins.append(margin)
    tail = linear.infer([r for r in rows if r['iteration'] >= end - 2], model, expected_ranks)
    if tail['status'] != 'localized' or tail['rank'] != rank:
        return None
    return dict(rank=rank, first_iteration=first, last_iteration=end,
                first_score=scores[first][rank], tail_scores=tail_scores, tail_margins=margins)
