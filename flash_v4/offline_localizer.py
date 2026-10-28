"""Exploratory rank ordering from complete sampled compute observations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from itertools import combinations
import math
from pathlib import Path
import statistics


EXPECTED_RANKS = tuple(range(16))
EXPECTED_ITERATIONS = 8
GEMM_KINDS = {"gemm", "lt_matmul"}
BRANCH_KINDS = {"gemm": GEMM_KINDS, "kernel": {"kernel"}}
HEAVY_FLOPS = 1_000_000_000
KERNEL_APIS = {
    "runtime", "runtime_ex", "runtime_ptsz", "runtime_ex_ptsz",
    "driver", "driver_ex", "driver_ptsz", "driver_ex_ptsz",
}

# CUDA data type enum widths used by the current observation corpus. Unknown
# values remain visible as uncovered events instead of being guessed.
DTYPE_BYTES = {0: 4, 1: 8, 2: 2, 14: 2}


class InputError(ValueError):
    pass


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _rank_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.name == "observations":
        files = sorted(root.glob("rank_*.jsonl"))
    else:
        files = sorted(root.glob("node*/observations/rank_*.jsonl"))
    if any(path.parent.name != "observations" for path in files):
        raise InputError("input must contain observation rank JSONL only")
    return files


def _read_observations(root: Path):
    files = _rank_files(root)
    by_rank = {}
    for path in files:
        try:
            rank = int(path.stem.removeprefix("rank_"))
        except ValueError:
            raise InputError("observation file has an invalid rank name") from None
        if rank in by_rank:
            raise InputError("duplicate observation file for a rank")
        rows = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        rows.append(json.loads(line))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise InputError("could not read an observation rank JSONL") from None
        summaries = [row for row in rows if row.get("kind") == "summary"]
        if len(summaries) != 1:
            raise InputError("each rank requires exactly one complete summary")
        summary = summaries[0]
        if (summary.get("rank") != rank
                or summary.get("window_complete") is not True
                or summary.get("profile_iterations") != EXPECTED_ITERATIONS
                or summary.get("completed_profile_iterations") != EXPECTED_ITERATIONS
                or summary.get("dropped") != 0):
            raise InputError("rank summary is incomplete or has dropped events")
        event_rows = [row for row in rows if row.get("kind") not in ("summary", "communicator")]
        if summary.get("emitted") != len(event_rows):
            raise InputError("summary event count does not match observation rows")
        kernel_rows = [row for row in event_rows if row.get("kind") == "kernel"]
        kernel_summary = summary.get("kernel")
        if kernel_rows:
            required_kernel_counts = (
                "candidates", "sampled", "emitted", "dropped", "skipped",
                "nccl", "unknown", "blas", "nested",
            )
            if not isinstance(kernel_summary, dict) or any(
                isinstance(kernel_summary.get(key), bool)
                or not isinstance(kernel_summary.get(key), int)
                or kernel_summary[key] < 0
                for key in required_kernel_counts
            ):
                raise InputError("kernel observation rows require complete nested kernel summary")
            if (kernel_summary.get("stride") != 256
                    or kernel_summary.get("id_hash") != "fnv1a64_device_name"
                    or kernel_summary.get("emitted") != len(kernel_rows)):
                raise InputError("kernel summary does not match the sampled observation rows")
        observed_iterations = {
            row.get("iteration") for row in event_rows
            if row.get("kind") in GEMM_KINDS
        }
        if len(observed_iterations) != EXPECTED_ITERATIONS or any(
            not isinstance(iteration, int) for iteration in observed_iterations
        ):
            raise InputError("rank does not contain eight GEMM observation iterations")
        by_rank[rank] = {
            "summary": summary,
            "rows": rows,
            "event_rows": event_rows,
            "iterations": tuple(sorted(observed_iterations)),
        }
    if tuple(sorted(by_rank)) != EXPECTED_RANKS:
        raise InputError("input requires one complete observation stream for ranks 0 through 15")
    iteration_sets = {item["iterations"] for item in by_rank.values()}
    if len(iteration_sets) != 1:
        raise InputError("rank observation iteration windows do not match")
    return by_rank


def _gemm_family_and_features(row, sampling_stride):
    kind = row.get("kind")
    if kind not in GEMM_KINDS:
        return None, None, "out_of_model_kind"
    if row.get("skip", ""):
        return None, None, "observer_skip"
    shape = row.get("shape")
    batch = row.get("batch")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                   for value in shape)
            or isinstance(batch, bool) or not isinstance(batch, int) or batch < 1):
        return None, None, "invalid_shape_or_batch"
    dtype_values = tuple(row.get(key) for key in ("dtype", "dtype_b", "dtype_c", "dtype_d"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in dtype_values):
        return None, None, "invalid_dtype"
    sizes = [DTYPE_BYTES.get(value) for value in dtype_values]
    if any(size is None for size in sizes):
        return None, None, "unsupported_dtype_width"
    compute = row.get("compute")
    op = row.get("op")
    if (isinstance(compute, bool) or not isinstance(compute, int)
            or isinstance(op, bool) or not isinstance(op, int)):
        return None, None, "invalid_compute_descriptor"

    m, n, k = shape
    flops = 2 * m * n * k * batch
    byte_count = batch * (
        m * k * sizes[0] + k * n * sizes[1] + m * n * (sizes[2] + sizes[3])
    )
    elapsed_us = _finite_number(row.get("elapsed_us"))
    if elapsed_us is None or elapsed_us <= 0:
        return None, None, "invalid_elapsed_time"
    family = (kind, *dtype_values, compute, op, batch)
    shape_key = (m, n, k)
    features = (1.0, flops / 1_000_000_000.0, byte_count / 1_000_000.0)
    sample_weight = 1.0 if flops >= HEAVY_FLOPS else float(sampling_stride)
    return (family, shape_key), {
        "family": family,
        "shape": shape_key,
        "features": features,
        "flops": flops,
        "elapsed_us": elapsed_us,
        "sample_weight": sample_weight,
    }, None


def _kernel_family_and_features(row):
    if row.get("kind") != "kernel":
        return None, None, "out_of_model_kind"
    stride = row.get("sampling_stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        return None, None, "invalid_row_sampling_stride"
    if row.get("skip", ""):
        return None, None, "kernel_skipped"
    kernel_id = row.get("kernel_id")
    if (not isinstance(kernel_id, str) or len(kernel_id) != 16
            or any(char not in "0123456789abcdefABCDEF" for char in kernel_id)):
        return None, None, "invalid_kernel_id"
    launch_api = row.get("launch_api")
    if launch_api not in KERNEL_APIS:
        return None, None, "invalid_launch_api"
    grid, block = row.get("grid"), row.get("block")
    def valid_dims(value):
        return (isinstance(value, list) and len(value) == 3
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                        for v in value))
    if not valid_dims(grid) or not valid_dims(block):
        return None, None, "invalid_grid_or_block"
    shared = row.get("dynamic_shared_bytes")
    if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
        return None, None, "invalid_dynamic_shared_bytes"
    elapsed_us = _finite_number(row.get("elapsed_us"))
    if elapsed_us is None or elapsed_us <= 0:
        return None, None, "invalid_elapsed_time"
    grid_blocks = math.prod(grid)
    block_threads = math.prod(block)
    family = (kernel_id.lower(), launch_api, block_threads)
    shape = (tuple(grid), tuple(block), shared)
    launched_threads = grid_blocks * block_threads
    return (family, shape), {
        "family": family,
        "shape": shape,
        "features": (1.0, float(launched_threads)),
        "grid_blocks": grid_blocks,
        "block_threads": block_threads,
        "elapsed_us": elapsed_us,
        "sample_weight": float(stride),
    }, None


def _branch_event_features(row, branch, summary):
    if branch == "gemm":
        return _gemm_family_and_features(row, _sampling_stride(summary))
    if branch == "kernel":
        return _kernel_family_and_features(row)
    raise InputError("unknown compute model branch")


def _sampling_stride(summary):
    stride = summary.get("stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise InputError("rank summary has an invalid sampling stride")
    return stride


def _solve_linear(matrix, vector):
    n = len(vector)
    augmented = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    if scale == 0.0:
        return None
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= scale * 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, n + 1):
            augmented[column][index] /= divisor
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, n + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[index][n] for index in range(n)]


def _weighted_nnls(features, observations, weights):
    if len(features) != len(observations) or len(features) != len(weights) or not features:
        raise InputError("insufficient shape samples for the compute model")
    parameter_count = len(features[0])
    if parameter_count not in (2, 3) or any(len(row) != parameter_count for row in features):
        raise InputError("compute model has an invalid feature width")
    scales = [max(abs(row[column]) for row in features) or 1.0
              for column in range(parameter_count)]
    normalized = [[row[column] / scales[column] for column in range(parameter_count)]
                  for row in features]
    best = None
    for mask in range(1 << parameter_count):
        active = [index for index in range(parameter_count) if mask & (1 << index)]
        if not active:
            coefficients = [0.0] * parameter_count
        else:
            normal = [[0.0 for _ in active] for _ in active]
            rhs = [0.0 for _ in active]
            for row, observed, weight in zip(normalized, observations, weights):
                if weight <= 0:
                    continue
                for left, col_left in enumerate(active):
                    rhs[left] += weight * row[col_left] * observed
                    for right, col_right in enumerate(active):
                        normal[left][right] += weight * row[col_left] * row[col_right]
            solution = _solve_linear(normal, rhs)
            if solution is None or any(value < -1e-10 for value in solution):
                continue
            coefficients = [0.0] * parameter_count
            for index, value in zip(active, solution):
                coefficients[index] = max(0.0, value) / scales[index]
        error = sum(
            weight * (sum(coef * x for coef, x in zip(coefficients, row)) - observed) ** 2
            for row, observed, weight in zip(features, observations, weights)
        )
        if best is None or error < best[0]:
            best = error, coefficients
    if best is None:
        raise InputError("compute feature matrix is numerically singular")
    return best[1]


def _robust_nnls(samples, rounds=3):
    features = [sample["features"] for sample in samples]
    observations = [sample["elapsed_us"] for sample in samples]
    parameter_count = len(features[0]) if features else 0
    minimum_shapes = max(3, parameter_count)
    if (parameter_count not in (2, 3) or len(samples) < minimum_shapes
            or len({sample["shape"] for sample in samples}) < minimum_shapes):
        raise InputError("insufficient distinct health shapes for a compute family")
    weights = [1.0] * len(samples)
    # Seed the IRLS loop with a small deterministic least-median subset search;
    # plain least squares can be pulled far off by one high-leverage outlier.
    ordered = sorted(range(len(samples)), key=lambda index: samples[index]["shape"])
    candidate_ids = []
    candidate_count = min(16, len(ordered))
    for position in range(candidate_count):
        selected = ordered[round(position * (len(ordered) - 1) / max(1, candidate_count - 1))]
        if selected not in candidate_ids:
            candidate_ids.append(selected)
    candidate_models = []
    for subset in combinations(candidate_ids, parameter_count):
        try:
            candidate = _weighted_nnls(
                [features[index] for index in subset],
                [observations[index] for index in subset],
                [1.0] * parameter_count,
            )
        except InputError:
            continue
        residuals = [abs(observed - sum(c * x for c, x in zip(candidate, row)))
                     for row, observed in zip(features, observations)]
        candidate_models.append((statistics.median(residuals), candidate))
    if not candidate_models:
        coefficients = _weighted_nnls(features, observations, weights)
    else:
        coefficients = min(candidate_models, key=lambda item: item[0])[1]
    for _ in range(rounds):
        residuals = [observed - sum(c * x for c, x in zip(coefficients, row))
                     for row, observed in zip(features, observations)]
        center = statistics.median(residuals)
        deviations = [abs(value - center) for value in residuals]
        mad = statistics.median(deviations)
        scale = 1.4826 * mad
        if scale <= 1e-12:
            maximum_deviation = max(deviations, default=0.0)
            if maximum_deviation <= 1e-12:
                break
            scale = maximum_deviation * 1e-6
        cutoff = 1.345 * scale
        weights = [1.0 if deviation <= cutoff else cutoff / deviation
                   for deviation in deviations]
        coefficients = _weighted_nnls(features, observations, weights)

    residuals = [observed - sum(c * x for c, x in zip(coefficients, row))
                 for row, observed in zip(features, observations)]
    center = statistics.median(residuals)
    deviations = [abs(value - center) for value in residuals]
    scale = 1.4826 * statistics.median(deviations)
    if scale <= 1e-12:
        maximum_deviation = max(deviations, default=0.0)
        if maximum_deviation <= 1e-12:
            weights = [1.0] * len(samples)
        else:
            scale = maximum_deviation * 1e-6
    if scale > 1e-12:
        cutoff = 1.345 * scale
        weights = [1.0 if deviation <= cutoff else cutoff / deviation
                   for deviation in deviations]
    return coefficients, weights


def _predict(coefficients, features):
    value = sum(coefficient * feature for coefficient, feature in zip(coefficients, features))
    return value if math.isfinite(value) and value > 0 else None


def _collect_health_samples(data, branch):
    by_family_shape_rank = defaultdict(list)
    unknown = Counter()
    for rank, content in data.items():
        for row in content["event_rows"]:
            if row.get("kind") not in BRANCH_KINDS[branch]:
                continue
            key, sample, reason = _branch_event_features(row, branch, content["summary"])
            if reason:
                unknown[reason] += 1
                continue
            family, shape = key
            by_family_shape_rank[(rank, family, shape)].append(sample)

    family_rows = defaultdict(list)
    for (rank, family, shape), samples in by_family_shape_rank.items():
        median_sample = dict(samples[0])
        median_sample["rank"] = rank
        median_sample["elapsed_us"] = statistics.median(
            sample["elapsed_us"] for sample in samples
        )
        family_rows[family].append(median_sample)

    models = {}
    model_info = []
    for family, samples in family_rows.items():
        distinct_shapes = {sample["shape"] for sample in samples}
        parameter_count = len(samples[0]["features"])
        if len(distinct_shapes) < max(3, parameter_count):
            unknown["family_insufficient_distinct_shapes"] += len(samples)
            continue
        coefficients, weights = _robust_nnls(samples)
        models[family] = coefficients
        coefficient_names = (
            ("intercept_us", "us_per_gflop", "us_per_megabyte")
            if branch == "gemm" else ("intercept_us", "us_per_launched_thread")
        )
        model_info.append({
            "family": list(family),
            "aggregated_rank_shape_samples": len(samples),
            "distinct_shapes": len(distinct_shapes),
            "robust_reweight_rounds": 3,
            "coefficients": dict(zip(coefficient_names, coefficients)),
            "robust_downweighted_shape_rows": sum(weight < 0.999 for weight in weights),
        })
    if not models and branch == "gemm":
        raise InputError("insufficient health GEMM shape families to fit the model")
    return models, model_info, unknown


def _score_dataset(data, models, branch):
    records = {rank: {} for rank in EXPECTED_RANKS}
    coverage = {}
    for rank, content in data.items():
        unknown_reasons = Counter()
        totals = {
            "event_count": 0,
            "known_event_count": 0,
            "unknown_event_count": 0,
            "inverse_probability_weighted_events": 0.0,
            "known_inverse_probability_weighted_events": 0.0,
            "unknown_inverse_probability_weighted_events": 0.0,
            "known_weighted_actual_us": 0.0,
            "unknown_weighted_actual_us": 0.0,
            "known_weighted_predicted_us": 0.0,
        }
        per_iteration = defaultdict(lambda: {
            "known_event_count": 0,
            "unknown_event_count": 0,
            "known_weighted_actual_us": 0.0,
            "unknown_weighted_actual_us": 0.0,
            "known_weighted_predicted_us": 0.0,
        })
        for row in content["event_rows"]:
            if row.get("kind") not in BRANCH_KINDS[branch]:
                continue
            totals["event_count"] += 1
            key, sample, reason = _branch_event_features(row, branch, content["summary"])
            if key is None:
                unknown_reasons[reason] += 1
                totals["unknown_event_count"] += 1
                continue
            iteration = row.get("iteration")
            if not isinstance(iteration, int):
                unknown_reasons["invalid_iteration"] += 1
                totals["unknown_event_count"] += 1
                continue
            weight = sample["sample_weight"]
            totals["inverse_probability_weighted_events"] += weight
            model = models.get(sample["family"])
            prediction = _predict(model, sample["features"]) if model is not None else None
            if prediction is None:
                unknown_reasons["unmodeled_family"] += 1
                totals["unknown_event_count"] += 1
                totals["unknown_inverse_probability_weighted_events"] += weight
                totals["unknown_weighted_actual_us"] += sample["elapsed_us"] * weight
                per_iteration[iteration]["unknown_event_count"] += 1
                per_iteration[iteration]["unknown_weighted_actual_us"] += (
                    sample["elapsed_us"] * weight
                )
                continue
            totals["known_event_count"] += 1
            totals["known_inverse_probability_weighted_events"] += weight
            totals["known_weighted_actual_us"] += sample["elapsed_us"] * weight
            totals["known_weighted_predicted_us"] += prediction * weight
            current = per_iteration[iteration]
            current["known_event_count"] += 1
            current["known_weighted_actual_us"] += sample["elapsed_us"] * weight
            current["known_weighted_predicted_us"] += prediction * weight
        coverage[rank] = {
            **totals,
            "model_coverage_by_weight": (
                totals["known_inverse_probability_weighted_events"]
                / totals["inverse_probability_weighted_events"]
                if totals["inverse_probability_weighted_events"] else 0.0
            ),
            "unknown_reasons": dict(sorted(unknown_reasons.items())),
        }
        records[rank] = dict(per_iteration)
    return records, coverage


def _health_offsets(health_data, models, branch):
    per_rank, coverage = _score_dataset(health_data, models, branch)
    offsets = {}
    for rank in EXPECTED_RANKS:
        ratios = []
        for values in per_rank[rank].values():
            predicted = values["known_weighted_predicted_us"]
            actual = values["known_weighted_actual_us"]
            if predicted > 0 and actual > 0:
                ratios.append(actual / predicted)
        if branch == "gemm" and len(ratios) != EXPECTED_ITERATIONS:
            raise InputError("health GEMM data has insufficient modeled samples per rank iteration")
        offset = statistics.median(ratios) if ratios else None
        if offset is not None and (not math.isfinite(offset) or offset <= 0):
            raise InputError("health rank multiplier could not be estimated")
        offsets[rank] = offset
    return offsets, coverage, per_rank


def _analyze_branch(healthy, condition, branch, iterations):
    models, model_info, health_fit_unknown = _collect_health_samples(healthy, branch)
    offsets, health_coverage, _ = _health_offsets(healthy, models, branch)
    condition_rows, condition_coverage = _score_dataset(condition, models, branch)
    raw_ratios = {rank: {} for rank in EXPECTED_RANKS}
    for rank in EXPECTED_RANKS:
        offset = offsets[rank]
        if offset is None:
            continue
        for iteration in iterations:
            values = condition_rows[rank].get(iteration)
            if (values and values["known_event_count"] > 0
                    and values["known_weighted_predicted_us"] > 0):
                raw_ratios[rank][iteration] = (
                    values["known_weighted_actual_us"]
                    / values["known_weighted_predicted_us"] / offset
                )
        if branch == "gemm" and len(raw_ratios[rank]) != EXPECTED_ITERATIONS:
            raise InputError("condition GEMM data has insufficient modeled samples per iteration")

    common_drift = {}
    normalized = {rank: {} for rank in EXPECTED_RANKS}
    for iteration in iterations:
        active = [raw_ratios[rank][iteration] for rank in EXPECTED_RANKS
                  if iteration in raw_ratios[rank]]
        if not active:
            continue
        drift = statistics.median(active)
        if not math.isfinite(drift) or drift <= 0:
            raise InputError("condition branch common drift could not be estimated")
        common_drift[iteration] = drift
        for rank in EXPECTED_RANKS:
            if iteration in raw_ratios[rank]:
                normalized[rank][iteration] = raw_ratios[rank][iteration] / drift

    top_counts = {rank: 0.0 for rank in EXPECTED_RANKS}
    for iteration in common_drift:
        active = {rank: normalized[rank][iteration] for rank in EXPECTED_RANKS
                  if iteration in normalized[rank]}
        if not active:
            continue
        maximum = max(active.values())
        winners = [rank for rank, value in active.items()
                   if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12)]
        for rank in winners:
            top_counts[rank] += 1.0 / len(winners)

    rankings = []
    for rank in EXPECTED_RANKS:
        per_iteration = {}
        for iteration in iterations:
            item = condition_rows[rank].get(iteration)
            if item is None:
                continue
            per_iteration[str(iteration)] = {
                **item,
                "raw_actual_over_prediction": (
                    item["known_weighted_actual_us"] / item["known_weighted_predicted_us"]
                    if item["known_weighted_predicted_us"] > 0 else None
                ),
                "health_offset_adjusted_ratio": raw_ratios[rank].get(iteration),
                "common_drift_normalized_score": normalized[rank].get(iteration),
            }
        scores = list(normalized[rank].values())
        rankings.append({
            "rank": rank,
            "score_median": statistics.median(scores) if scores else None,
            "score_by_iteration": {str(key): value
                                   for key, value in normalized[rank].items()},
            "iterations_scored": len(scores),
            "slowest_rank_iteration_wins": top_counts[rank],
            "health_time_multiplier": offsets[rank],
            "model_coverage": condition_coverage[rank],
            "raw_condition_totals": condition_coverage[rank],
            "per_iteration_raw_and_normalized": per_iteration,
        })
    rankings.sort(key=lambda item: (
        item["score_median"] is None,
        -(item["score_median"] or 0.0),
        item["rank"],
    ))
    return {
        "branch": branch,
        "model_status": "fitted" if models else "unavailable_or_insufficient_health_shapes",
        "models": model_info,
        "unfit_health_family_rows": dict(sorted(health_fit_unknown.items())),
        "health_time_multiplier_by_rank": {str(k): v for k, v in offsets.items()},
        "health_model_coverage_by_rank": {str(k): v for k, v in health_coverage.items()},
        "condition_model_coverage_by_rank": {
            str(k): v for k, v in condition_coverage.items()
        },
        "common_drift_by_iteration": {str(k): v for k, v in common_drift.items()},
        "rank_order": rankings,
        "_records": condition_rows,
        "_offsets": offsets,
        "_raw_ratios": raw_ratios,
        "_normalized": normalized,
    }


def _combine_branches(branches, iterations):
    gemm, kernel = branches["gemm"], branches["kernel"]
    raw = {rank: {} for rank in EXPECTED_RANKS}
    quantities = {rank: {} for rank in EXPECTED_RANKS}
    for rank in EXPECTED_RANKS:
        for iteration in iterations:
            branch_items = []
            for branch in (gemm, kernel):
                values = branch["_records"][rank].get(iteration)
                offset = branch["_offsets"][rank]
                if (values is None or offset is None or values["known_event_count"] < 1
                        or values["known_weighted_predicted_us"] <= 0):
                    branch_items = []
                    break
                branch_items.append((values, offset))
            if len(branch_items) != 2:
                continue
            actual = sum(item["known_weighted_actual_us"] for item, _ in branch_items)
            calibrated_prediction = sum(
                item["known_weighted_predicted_us"] * offset
                for item, offset in branch_items
            )
            if actual <= 0 or calibrated_prediction <= 0:
                continue
            raw[rank][iteration] = actual / calibrated_prediction
            quantities[rank][iteration] = {
                "inverse_probability_weighted_actual_us": actual,
                "health_offset_calibrated_predicted_us": calibrated_prediction,
                "gemm_actual_us": branch_items[0][0]["known_weighted_actual_us"],
                "kernel_actual_us": branch_items[1][0]["known_weighted_actual_us"],
                "gemm_calibrated_predicted_us": (
                    branch_items[0][0]["known_weighted_predicted_us"] * branch_items[0][1]
                ),
                "kernel_calibrated_predicted_us": (
                    branch_items[1][0]["known_weighted_predicted_us"] * branch_items[1][1]
                ),
            }

    common_drift = {}
    normalized = {rank: {} for rank in EXPECTED_RANKS}
    for iteration in iterations:
        active = [raw[rank][iteration] for rank in EXPECTED_RANKS
                  if iteration in raw[rank]]
        if not active:
            continue
        drift = statistics.median(active)
        common_drift[iteration] = drift
        for rank in EXPECTED_RANKS:
            if iteration in raw[rank]:
                normalized[rank][iteration] = raw[rank][iteration] / drift

    wins = {rank: 0.0 for rank in EXPECTED_RANKS}
    for iteration in common_drift:
        active = {rank: normalized[rank][iteration] for rank in EXPECTED_RANKS
                  if iteration in normalized[rank]}
        maximum = max(active.values())
        winners = [rank for rank, value in active.items()
                   if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12)]
        for rank in winners:
            wins[rank] += 1.0 / len(winners)

    ranking = []
    for rank in EXPECTED_RANKS:
        scores = list(normalized[rank].values())
        ranking.append({
            "rank": rank,
            "score_median": statistics.median(scores) if scores else None,
            "score_by_iteration": {str(k): v for k, v in normalized[rank].items()},
            "iterations_with_both_branches": len(scores),
            "slowest_rank_iteration_wins": wins[rank],
            "per_iteration_raw_totals": {
                str(k): {
                    **quantities[rank][k],
                    "raw_total_actual_over_calibrated_prediction": raw[rank][k],
                    "common_drift_normalized_score": normalized[rank][k],
                }
                for k in quantities[rank]
            },
        })
    ranking.sort(key=lambda item: (
        item["score_median"] is None,
        -(item["score_median"] or 0.0),
        item["rank"],
    ))
    return {
        "method": "per rank and iteration: sum of both branches' inverse-probability-weighted observed microseconds divided by the sum of branch-specific health-offset-calibrated predicted microseconds; then divide by the median across ranks for that iteration",
        "scope": "non-overlapping sampled library and launch call envelopes; not total GPU utilization or wall-clock time",
        "time_weighting": "implicit from predicted covered duration; no branch weight is selected from truth",
        "rank_order": ranking,
        "common_drift_by_iteration": {str(k): v for k, v in common_drift.items()},
        "combined_available_for_any_rank_iteration": bool(common_drift),
    }


def analyze(healthy_root: Path, condition_root: Path):
    healthy = _read_observations(healthy_root)
    condition = _read_observations(condition_root)
    if healthy[0]["iterations"] != condition[0]["iterations"]:
        raise InputError("healthy and condition GEMM iteration windows do not match")
    iterations = list(condition[0]["iterations"])
    branch_results = {
        branch: _analyze_branch(healthy, condition, branch, iterations)
        for branch in ("gemm", "kernel")
    }
    combined = _combine_branches(branch_results, iterations)

    rank_rows = []
    branch_by_rank = {
        branch: {row["rank"]: row for row in result["rank_order"]}
        for branch, result in branch_results.items()
    }
    combined_by_rank = {row["rank"]: row for row in combined["rank_order"]}
    for rank in EXPECTED_RANKS:
        rank_rows.append({
            "rank": rank,
            "gemm_score": branch_by_rank["gemm"][rank]["score_median"],
            "kernel_score": branch_by_rank["kernel"][rank]["score_median"],
            "combined_total_ratio_score": combined_by_rank[rank]["score_median"],
            "combined_iterations_with_both_branches": (
                combined_by_rank[rank]["iterations_with_both_branches"]
            ),
        })
    rank_rows.sort(key=lambda item: (
        item["combined_total_ratio_score"] is None,
        -(item["combined_total_ratio_score"] or 0.0),
        item["rank"],
    ))

    summaries = [data[rank]["summary"] for data in (healthy, condition)
                 for rank in EXPECTED_RANKS]
    kernel_summary_counts = {
        "healthy": {str(rank): healthy[rank]["summary"].get("kernel") for rank in EXPECTED_RANKS},
        "condition": {str(rank): condition[rank]["summary"].get("kernel") for rank in EXPECTED_RANKS},
    }
    for result in branch_results.values():
        result.pop("_records")
        result.pop("_offsets")
        result.pop("_raw_ratios")
        result.pop("_normalized")
    return {
        "schema": "flash_v4_offline_localizer_exploration_v2",
        "analysis_type": "exploratory_rank_ordering_only",
        "formal_acceptance_eligible": False,
        "localization_passed": False,
        "accuracy_reported": False,
        "score_scope": "sampled GemmEx/LtMatmul library-call and sampled CUDA kernel-launch envelopes; not pure kernel time",
        "inputs": {
            "healthy_observations": str(healthy_root),
            "condition_observations": str(condition_root),
        },
        "branches": branch_results,
        "combined_total_ratio": combined,
        "rank_order": rank_rows,
        "sampling": {
            "gemm_heavy_flops_threshold": HEAVY_FLOPS,
            "gemm_heavy_event_weight": 1,
            "gemm_light_event_weight": "per-rank summary stride",
            "kernel_event_weight": "per-row sampling_stride",
            "weight_interpretation": "inverse-probability sampling estimate",
        },
        "models": {
            "gemm": {
                "family_fields": ["kind", "dtype", "dtype_b", "dtype_c", "dtype_d",
                                  "compute", "op", "batch"],
                "features": ["intercept", "FLOPs/1e9", "estimated bytes/1e6"],
                "estimated_bytes_formula": "batch*(m*k*size(dtype)+k*n*size(dtype_b)+m*n*(size(dtype_c)+size(dtype_d)))",
            },
            "kernel": {
                "family_fields": ["kernel_id", "launch_api", "block_threads"],
                "features": ["intercept", "grid_blocks*block_threads"],
                "estimated_time_formula": "intercept + slope*grid_blocks*block_threads",
                "shape_median_fields": ["grid", "block", "dynamic_shared_bytes"],
                "row_declared_sampling_stride": 256,
            },
            "fit": "health shape medians; nonnegative NNLS by active-set enumeration; 3 robust reweight rounds",
        },
        "kernel_summary_counters": kernel_summary_counts,
        "data_quality": {
            "ranks": 16,
            "profile_iterations_per_rank": EXPECTED_ITERATIONS,
            "iteration_ids": iterations,
            "all_summaries_complete_and_zero_top_level_drops": True,
            "summary_coverage_declarations": sorted({
                str(summary.get("coverage")) for summary in summaries
            }),
            "warning": "observation coverage is limited; this exploratory ordering is not an acceptance result",
        },
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = analyze(args.healthy, args.condition)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (InputError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
