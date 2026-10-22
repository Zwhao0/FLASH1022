"""FLASH Ch.3 analysis; preprocessing/BEAST flow adapted from Greyhound-master/
detector/control_plane/slow_detection.py. No ACF inference, retry loop or silent fallback.
Rbeast's beast.py documents start=1,deltat=1; returned cp is a time, hence index=cp-1.
"""
from collections import defaultdict, deque
import math
import re
import statistics as st


def profile_seconds(iter_ms):
    if not math.isfinite(iter_ms) or iter_ms <= 0:
        raise ValueError('iteration duration must be positive finite milliseconds')
    return max(iter_ms / 1000.0 * 2.2, 20.0)


def _cv(xs):
    return st.pstdev(xs) / st.mean(xs) if len(xs) > 1 else 0.0


def _beast_candidates(values):
    import numpy as np
    import Rbeast as rb
    result = rb.beast(np.asarray(values, dtype=float), start=1, deltat=1,
                      season='none', hasOutlier=True, mcmc_seed=42,
                      torder_minmax=[0, 0], tseg_minlength=2,
                      tseg_leftmargin=2, tseg_rightmargin=1,
                      print_param=False, print_progress=False, quiet=True)
    if not hasattr(result, 'trend'):
        raise RuntimeError('Rbeast returned no trend result')
    # cp entries are ranked by probability, not temporal order. Finite entries only.
    return [(int(round(float(cp))) - 1, float(prob))
            for cp, prob in zip(np.ravel(result.trend.cp), np.ravel(result.trend.cpPr))
            if math.isfinite(float(cp)) and math.isfinite(float(prob))]


class WindowDetector:
    """Paper 3.2.2 fixes monitoring at 30 steps; 3.2.3 adapts the profiling
    time boundary via profile_seconds, rather than changing this deque length.
    """
    def __init__(self, window=30, warmup=3, probability=.8, degradation=.2,
                 recovery=.75, right_guard=2, candidate_fn=None):
        if window < 6 or right_guard < 1:
            raise ValueError('window >=6 and right_guard >=1 required')
        self.window, self.warmup = window, warmup
        self.probability, self.degradation = probability, degradation
        self.recovery, self.right_guard = recovery, right_guard
        self.samples = deque(maxlen=window)
        self.candidate_fn = candidate_fn or _beast_candidates
        self.last_iteration = None
        self.last_cp = None

    def add(self, iteration, time_ms):
        if not math.isfinite(time_ms) or time_ms <= 0:
            raise ValueError('time_ms must be positive finite')
        if self.last_iteration is not None and iteration <= self.last_iteration:
            raise ValueError('iterations must strictly increase')
        self.last_iteration = iteration
        if iteration <= self.warmup:
            return None
        self.samples.append((iteration, float(time_ms)))
        if len(self.samples) < self.window:
            return None
        raw = list(self.samples)
        vals = [v for _, v in raw]
        sd, mean = st.pstdev(vals), st.mean(vals)
        points = [p for p in raw if sd == 0 or abs(p[1] - mean) / sd <= 3]
        if len(points) < 6 or st.pstdev([p[1] for p in points]) == 0:
            return None
        values = [v for _, v in points]
        candidates = sorted((i, p) for i, p in self.candidate_fn(values)
                            if p >= self.probability and 1 <= i < len(points))
        dedup = []
        for i, p in candidates:
            cp = points[i][0]
            if not dedup or cp - points[dedup[-1][0]][0] >= 2:
                dedup.append((i, p))
        for position, (idx, prob) in enumerate(dedup):
            if len(points) - idx - 1 < self.right_guard:
                continue
            cp = points[idx][0]
            if self.last_cp is not None and cp - self.last_cp < 2:
                continue
            left = dedup[position - 1][0] if position else 0
            right = dedup[position + 1][0] if position + 1 < len(dedup) else len(points)
            xl, xr = values[left:idx], values[idx:right]
            if not xl or len(xr) < self.right_guard + 1:
                continue
            baseline, degraded, recent = st.mean(xl), st.mean(xr), st.mean(xr[1:])
            drop = (degraded - baseline) / baseline
            if drop <= self.degradation:
                continue
            ratio = (degraded - recent) / (degraded - baseline)
            if ratio >= self.recovery:
                continue
            self.last_cp = cp
            return dict(cp=cp, cp_index=idx, probability=prob, recovery_ratio=ratio,
                        baseline_mean=baseline, degraded_mean=degraded,
                        recent_mean=recent, degradation=drop, iteration=iteration,
                        window_start=raw[0][0], window_end=raw[-1][0],
                        filtered_iterations=[it for it, _ in raw if it not in {p[0] for p in points}],
                        profile_seconds=profile_seconds(time_ms), method='Rbeast')
        return None


def _profile_key(row):
    # Legacy tags encoded observation shape; shape is never a semantic group.
    tag = str(row['tag'])
    family, count, dtype = tag, row.get('count'), row.get('dtype')
    match = re.fullmatch(r'(.*):count([^:]+):dtype([^:]+)', tag)
    if match:
        family = match[1]
        count = row.get('count', match[2])
        dtype = row.get('dtype', match[3])
    return family, None if count is None else str(count), None if dtype is None else str(dtype)


def localize(rows, cv_threshold=.1, rank_threshold=.2, topology=None, baseline_rows=None):
    """Compare equal family/count/dtype observations, then give each real group one vote.

    Pipeline candidates need comparable cross-stage screening to distinguish the
    source from backpressure. Membership comes from reported topology, never IDs.
    """
    rows = list(rows)
    if baseline_rows is not None:
        raise ValueError('Healthy PROFILE differential localization is temporarily disabled; use no reference profile')
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    expected_members = defaultdict(set)
    expected_sizes = defaultdict(set)
    for row in rows:
        key = (_profile_key(row)[0], str(row['clique']))
        expected_members[key].update(row.get('group_members', [int(row['rank'])]))
        if row.get('num_devices') is not None:
            expected_sizes[key].add(int(row['num_devices']))
    for row in rows:
        value = float(row['duration_us'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('duration_us must be positive finite')
        clique, rank = str(row['clique']), int(row['rank'])
        comm = str(row.get('comm', row.get('comm_addr', rank)))
        groups[_profile_key(row)][clique][(rank, comm)].append(value)
    evidence, skipped = [], []
    for (family, count, dtype), cliques in sorted(groups.items(), key=lambda pair: str(pair[0])):
        detail = dict(tag=family, family=family, count=count, dtype=dtype)
        valid = {}
        for c, data in cliques.items():
            members = {r for r, _ in data}
            sizes = expected_sizes[family, c]
            if (members != expected_members[family, c]
                    or (sizes and sizes != {len(members)})):
                skipped.append(dict(detail, clique=c, members=sorted(members),
                                    reason='incomplete shape observations'))
            elif len(members) > 1:
                valid[c] = data
        if not valid:
            skipped.append(dict(detail, reason='no multi-rank clique'))
            continue
        comparison = [dict(clique=c, members=sorted({r for r, _ in data}))
                      for c, data in sorted(valid.items())]
        detail.update(comparison_cliques=comparison, group_screened=len(valid) > 1)
        maxima = {c: max(st.mean(v) for v in data.values()) for c, data in valid.items()}
        group_cv = _cv(list(maxima.values()))
        if len(valid) > 1 and group_cv < cv_threshold:
            skipped.append(dict(detail, reason='clique CV below threshold', cv=group_cv))
            continue
        clique = max(sorted(valid), key=lambda c: maxima[c])
        by_rank = defaultdict(list)
        for (rank, _), observations in valid[clique].items():
            by_rank[rank].extend(observations)
        averages = {r: st.mean(v) for r, v in by_rank.items()}
        rank_cv = _cv(list(averages.values()))
        mean = st.mean(averages.values())
        rank = min(sorted(averages), key=averages.get)
        score = (mean - averages[rank]) / mean
        detail.update(clique=clique, members=sorted(averages))
        if rank_cv < cv_threshold or score < rank_threshold:
            skipped.append(dict(detail, reason='insufficient rank separation', cv=rank_cv, score=score))
            continue
        evidence.append(dict(detail, rank=rank, rank_means_us=averages,
                             clique_max_avg_us=maxima[clique], clique_cv=group_cv,
                             rank_cv=rank_cv, score=score))

    real_groups = defaultdict(list)
    for item in evidence:
        real_groups[item['family'], item['clique']].append(item)
    candidates = defaultdict(lambda: dict(tags=[], groups=[], score=0.0))
    summaries = []
    raw_scores = defaultdict(float)
    for (family, clique), items in sorted(real_groups.items()):
        ranks = sorted({item['rank'] for item in items})
        conflict = len(ranks) > 1
        summary = dict(family=family, clique=clique, candidates=ranks, conflict=conflict,
                       rank=None if conflict else ranks[0],
                       score=0.0 if conflict else st.mean(item['score'] for item in items))
        summaries.append(summary)
        for rank in ranks:
            raw_scores[rank] += st.mean(item['score'] for item in items if item['rank'] == rank)
            candidate = candidates[rank]
            if family not in candidate['tags']:
                candidate['tags'].append(family)
            candidate['groups'].append(summary)
            if not conflict:
                candidate['score'] += summary['score']
    ordered = sorted(candidates, key=lambda r: (-candidates[r]['score'], -len(candidates[r]['tags']), r))
    winner = ordered[0] if ordered and candidates[ordered[0]]['score'] > 0 else None
    raw_rank = min(raw_scores, key=lambda r: (-raw_scores[r], r)) if raw_scores else None
    reason = None
    if any(group['conflict'] for group in summaries):
        winner = None
        reason = 'conflicting sizes within a semantic group; no certain localization'
    topology_groups = (topology or {}).get('groups', [])
    pipelines = [set(g['members']) for g in topology_groups if g.get('name') == 'pipeline']
    for row in rows:
        members = row.get('pipeline_members')
        if members:
            pipelines.append(set(members))
    stages = [set(g['members']) for g in topology_groups
              if g.get('name') == 'dense_DP' or g.get('label') == 'dense_DP']
    if pipelines and len(ordered) > 1:
        same_chain = any(len(chain.intersection(ordered)) > 1 for chain in pipelines)
        same_stage = any(set(ordered).issubset(stage) for stage in stages)
        if same_chain or not same_stage:
            # Only equal-shape stage-1 comparisons may resolve this ambiguity.
            screened = [item for item in evidence if item['rank'] == winner and item['group_screened']
                        and set(ordered).issubset({r for c in item['comparison_cliques'] for r in c['members']})
                        and not next(g['conflict'] for g in summaries
                                     if g['family'] == item['family'] and g['clique'] == item['clique'])]
            if not screened:
                winner = None
                reason = ('pipeline chain has conflicting candidates; no comparable cross-stage screening'
                          if same_chain else
                          'candidates span pipeline stages; no comparable cross-stage screening')
    if winner is None and reason is None:
        reason = 'conflicting sizes within a semantic group' if ordered else 'insufficient evidence'
    result = dict(rank=winner, raw_rank=raw_rank, status='identified' if winner is not None else 'inconclusive',
                  candidates=[dict(rank=r, **candidates[r]) for r in ordered],
                  groups=summaries, evidence=evidence, skipped=skipped)
    if reason:
        result['reason'] = reason
    return result


def _reference_localize(rows, baseline_rows, rank_threshold, uncalibrated):
    """Subtract a matched healthy PROFILE before comparing added peer waiting.

    Counts/operation stages are never pooled as raw durations. Subtraction is
    per shape, actual membership and rank; each real group then has one vote.
    This calibrated extension is distinct from the paper's absolute-latency rule.
    """
    def collect(observations):
        members = defaultdict(set)
        declared = defaultdict(set)
        sizes = defaultdict(set)
        for row in observations:
            clique = str(row['clique'])
            members[clique].add(int(row['rank']))
            if row.get('group_members'):
                declared[clique].add(tuple(sorted(row['group_members'])))
            if row.get('num_devices') is not None:
                sizes[clique].add(int(row['num_devices']))
        output = defaultdict(lambda: defaultdict(list))
        for row in observations:
            value = float(row['duration_us'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('duration_us must be positive finite')
            clique = str(row['clique'])
            observed = tuple(sorted(members[clique]))
            if (declared[clique] and declared[clique] != {observed}) or (sizes[clique] and sizes[clique] != {len(observed)}):
                continue
            key = (_profile_key(row), observed)
            output[key][int(row['rank'])].append(value)
        return output

    current, baseline = collect(rows), collect(baseline_rows)
    evidence, skipped = [], []
    for key, per_rank in sorted(current.items(), key=lambda pair: str(pair[0])):
        (family, count, dtype), members = key
        detail = dict(family=family, tag=family, count=count, dtype=dtype, members=list(members))
        reference = baseline.get(key)
        if (len(members) < 2 or reference is None or set(per_rank) != set(members)
                or set(reference) != set(members)
                or any(len(v) < 2 for v in per_rank.values())
                or any(len(v) < 2 for v in reference.values())):
            skipped.append(dict(detail, reason='missing matched healthy samples'))
            continue
        means = {r: st.mean(v) for r, v in per_rank.items()}
        healthy = {r: st.mean(v) for r, v in reference.items()}
        delta = {r: means[r] - healthy[r] for r in members}
        noise = 3 * max(math.sqrt(st.variance(per_rank[r]) / len(per_rank[r]) +
                                  st.variance(reference[r]) / len(reference[r])) for r in members)
        signal = max(delta.values()) - min(delta.values())
        gate = max(noise, rank_threshold * st.mean(healthy.values()))
        if signal <= gate:
            skipped.append(dict(detail, reason='added waiting below reference/noise gate',
                                signal_us=signal, gate_us=gate))
            continue
        minima = [r for r in members if delta[r] == min(delta.values())]
        if len(minima) != 1:
            skipped.append(dict(detail, reason='reference-adjusted minimum tied', candidates=minima))
            continue
        rank = minima[0]
        evidence.append(dict(detail, rank=rank, rank_means_us=means,
                             baseline_means_us=healthy, delta_us=delta,
                             samples_per_rank={r: len(v) for r, v in per_rank.items()},
                             baseline_samples_per_rank={r: len(v) for r, v in reference.items()},
                             score=signal, score_units='microseconds_added_peer_wait',
                             gate_us=gate, noise_us=noise))

    by_group = defaultdict(list)
    for item in evidence:
        by_group[item['family'], tuple(item['members'])].append(item)
    candidates = defaultdict(lambda: dict(tags=[], groups=[], score=0.0))
    summaries = []
    for (family, members), items in sorted(by_group.items()):
        votes = sorted({item['rank'] for item in items})
        summary = dict(family=family, members=list(members), candidates=votes,
                       conflict=len(votes) > 1, rank=votes[0] if len(votes) == 1 else None,
                       score=st.mean(item['score'] for item in items))
        summaries.append(summary)
        if len(votes) != 1:
            for rank in votes:
                candidates[rank]['groups'].append(summary)
                if family not in candidates[rank]['tags']:
                    candidates[rank]['tags'].append(family)
            continue
        candidate = candidates[votes[0]]
        candidate['score'] += summary['score']
        candidate['groups'].append(summary)
        if family not in candidate['tags']:
            candidate['tags'].append(family)
    ordered = sorted(candidates, key=lambda r: (-candidates[r]['score'], -len(candidates[r]['tags']), r))
    # Conflicting dimensions are never silently discarded in favor of weak votes.
    winner = ordered[0] if ordered and not any(g['conflict'] for g in summaries) else None
    result = dict(rank=winner, status='identified' if winner is not None else 'inconclusive',
                  method='healthy_profile_added_wait', score_units='microseconds_added_peer_wait',
                  candidates=[dict(rank=r, **candidates[r]) for r in ordered], groups=summaries,
                  evidence=evidence, skipped=skipped, uncalibrated=uncalibrated)
    if winner is None:
        result['reason'] = 'conflicting reference-adjusted sizes' if summaries else 'no significant matched-reference evidence'
    return result
