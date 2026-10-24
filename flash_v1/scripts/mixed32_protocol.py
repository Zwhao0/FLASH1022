"""CPU-only design, pilot calibration, and frozen-plan helpers for mixed32."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


LAYOUTS = ("legacy", "node_local_pp")
MODES = (
    "attention_bwd_sleep",
    "experts_bwd_sleep",
    "attention_fwd_sleep",
    "attention_bwd_compute",
)
TIER_RANGES = ((35.0, 45.0), (45.0, 55.0), (55.0, 65.0), (65.0, 80.0))
BALANCE_DIMS = {
    "legacy_pp_or_node": lambda rank: rank // 8,
    "node_local_pp": lambda rank: (rank // 4) % 2,
    "tp_member": lambda rank: rank % 2,
    "ep_member": lambda rank: (rank // 2) % 2,
}
MAX_ASSIGNMENT_ATTEMPTS = 100_000
PILOT_DELAY_MS_PER_MICROBATCH = 200.0
MICROBATCHES = 2
FORMAL_DELAY_MIN_MS = 60.0
FORMAL_DELAY_MAX_MS = 650.0

# These calibration gates are fixed protocol constants. They reject measurements
# whose timing spread obscures the injected signal; they do not alter classifier thresholds.
MAX_SAMPLE_CV = 0.25
MIN_SIGNAL_TO_NOISE = 2.0


def _check_layout(layout):
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def _pp_stage(layout, rank):
    _check_layout(layout)
    if not 0 <= int(rank) < 16:
        raise ValueError(f"rank must be in 0..15, got {rank!r}")
    rank = int(rank)
    return rank // 8 if layout == "legacy" else (rank // 4) % 2


def _balanced_assignment(modes_by_rank):
    if Counter(modes_by_rank) != Counter({mode: 4 for mode in MODES}):
        return False
    for coordinate in BALANCE_DIMS.values():
        for mode in MODES:
            sides = [sum(modes_by_rank[rank] == mode and coordinate(rank) == side
                         for rank in range(16)) for side in (0, 1)]
            if sides != [2, 2]:
                return False
    return True


def _validate_design(design):
    cases = design.get("cases") if isinstance(design, dict) else None
    if not isinstance(cases, list) or len(cases) != 16:
        raise ValueError("design must contain exactly 16 rank cases")
    if sorted(case.get("rank") for case in cases) != list(range(16)):
        raise ValueError("design must assign exactly one case to every rank")
    modes_by_rank = [None] * 16
    mode_cases = {mode: [] for mode in MODES}
    for case in cases:
        rank, mode = case.get("rank"), case.get("fault_mode")
        if mode not in MODES:
            raise ValueError(f"unsupported design mode for rank {rank}: {mode!r}")
        modes_by_rank[rank] = mode
        mode_cases[mode].append(case)
        tier = case.get("tier")
        if tier not in range(4):
            raise ValueError(f"invalid tier for rank {rank}: {tier!r}")
        low, high = TIER_RANGES[tier]
        target = case.get("target_degradation_pct")
        if not isinstance(target, (int, float)) or not low <= target <= high:
            raise ValueError(f"rank {rank} target is outside tier {tier}: {target!r}")
    if not _balanced_assignment(modes_by_rank):
        raise ValueError("design does not satisfy the four binary balance constraints")
    for mode, rows in mode_cases.items():
        if len(rows) != 4 or {row["tier"] for row in rows} != {0, 1, 2, 3}:
            raise ValueError(f"mode {mode} must have one case in every tier")


def build_design(seed):
    """Create a paired 16-rank assignment with four balanced modes and tiers."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    assignment = None
    attempts = 0
    for attempts in range(1, MAX_ASSIGNMENT_ATTEMPTS + 1):
        shuffled = [mode for mode in MODES for _ in range(4)]
        rng.shuffle(shuffled)
        if _balanced_assignment(shuffled):
            assignment = shuffled
            break
    if assignment is None:
        raise RuntimeError(f"no balanced assignment found in {MAX_ASSIGNMENT_ATTEMPTS} attempts")

    cases = []
    for mode in MODES:
        ranks = [rank for rank, assigned_mode in enumerate(assignment) if assigned_mode == mode]
        tiers = [0, 1, 2, 3]
        rng.shuffle(tiers)
        for rank, tier in zip(ranks, tiers):
            low, high = TIER_RANGES[tier]
            target = round(rng.uniform(low, high), 3)
            cases.append({
                "case_id": f"rank{rank:02d}_{mode}",
                "rank": rank,
                "fault_mode": mode,
                "tier": tier,
                "target_degradation_pct": target,
                "tier_range_pct": [low, high],
            })
    cases.sort(key=lambda row: row["rank"])
    design = {
        "schema_version": 1,
        "protocol": "mixed32",
        "seed": seed,
        "assignment_attempts": attempts,
        "cases_per_layout": 16,
        "total_formal_cases": 32,
        "pairing": "same_rank_mode_tier_and_target_across_layouts",
        "balanced_dimensions": list(BALANCE_DIMS),
        "cases": cases,
    }
    _validate_design(design)
    return design


def _case_order_seed(seed, layout):
    salt = 0x9E3779B1 if layout == "legacy" else 0x85EBCA77
    return (seed ^ salt) & ((1 << 64) - 1)


def _make_interval(interval_id, start, length, **fields):
    row = {
        "id": interval_id,
        "start": start,
        "end": start + length - 1,
        "record_start": start + 1,
        "target_ranks": [],
        "delay_ms": 0.0,
        "fault_mode": None,
    }
    row.update(fields)
    return row


def build_pilot(session, layout, seed, delay_ms=PILOT_DELAY_MS_PER_MICROBATCH):
    """Build an offline 136-step pilot with eight paired mode/stage injections."""
    _check_layout(layout)
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (isinstance(delay_ms, bool) or not isinstance(delay_ms, (int, float))
            or not math.isfinite(delay_ms) or float(delay_ms) not in (200.0, 400.0)):
        raise ValueError("pilot delay_ms must be exactly 200 or 400 per microbatch")
    pilot_delay = float(delay_ms)
    intervals = []
    start = 1

    warmup = _make_interval("warmup", start, 12, warmup=True, segment="warmup")
    intervals.append(warmup)
    start = warmup["end"] + 1
    opening = _make_interval("healthy_opening", start, 12, segment="healthy")
    intervals.append(opening)
    start = opening["end"] + 1

    probes = [(mode, pp_stage, 0 if pp_stage == 0 else 15)
              for mode in MODES for pp_stage in (0, 1)]
    random.Random(_case_order_seed(seed, layout)).shuffle(probes)
    for index, (mode, pp_stage, rank) in enumerate(probes, 1):
        healthy_id = f"healthy_before_case{index:02d}"
        healthy = _make_interval(
            healthy_id, start, 8, segment="healthy", calibration_reference=True,
            case_index=index, fault_mode=mode, pp_stage=pp_stage, pilot_rank=rank,
        )
        intervals.append(healthy)
        start = healthy["end"] + 1
        fault = _make_interval(
            f"pilot_case{index:02d}_{mode}_pp{pp_stage}", start, 6,
            segment="fault", case_index=index, case_id=f"pilot_{mode}_pp{pp_stage}",
            target_ranks=[rank], delay_ms=pilot_delay,
            fault_mode=mode, pp_stage=pp_stage, pilot_rank=rank,
            baseline_interval_id=healthy_id,
            delay_scope="milliseconds_per_microbatch_selected_phase",
            microbatches=MICROBATCHES,
            requested_total_ms_per_iteration=pilot_delay * MICROBATCHES,
        )
        intervals.append(fault)
        start = fault["end"] + 1

    total_steps = start - 1
    if total_steps != 136:
        raise AssertionError(f"pilot duration changed unexpectedly: {total_steps}")
    return {
        "schema_version": 1,
        "session_id": session,
        "role": "mixed32_pilot",
        "layout": layout,
        "parallel_layout": layout,
        "seed": seed,
        "steps": total_steps,
        "warmup_steps": 12,
        "opening_healthy_steps": 12,
        "pilot_delay_ms_per_microbatch": pilot_delay,
        "online": False,
        "probe_enabled": False,
        "profile_enabled": False,
        "max_profiles": 0,
        "seq_len": 16,
        "microbatches": MICROBATCHES,
        "fault_kind": "mixed_compute_simulation",
        "measurement_only": True,
        "calibration_input": "host2/train.jsonl wall_iteration_ms only",
        "intervals": intervals,
    }


def _session_paths(pilot_session_path):
    path = Path(pilot_session_path)
    if path.is_file() and path.name == "train.jsonl":
        train_path = path
        session_root = path.parent.parent
    else:
        session_root = path
        train_path = session_root / "host2" / "train.jsonl"
    plan_path = session_root / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"pilot plan is missing: {plan_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"pilot timing log is missing: {train_path}")
    return session_root, plan_path, train_path


def _read_timing_rows(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                iteration = int(row.get("iteration", row.get("step")))
                wall_ms = float(row["wall_iteration_ms"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid timing row {line_number} in {path}: {error}") from error
            if iteration in rows:
                raise ValueError(f"duplicate timing iteration {iteration} in {path}")
            if not math.isfinite(wall_ms) or wall_ms <= 0:
                raise ValueError(f"invalid wall_iteration_ms at iteration {iteration}: {wall_ms}")
            rows[iteration] = wall_ms
    return rows


def _sample_summary(values):
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean_ms": mean, "sample_stddev_ms": deviation,
            "coefficient_of_variation": deviation / mean if mean else math.inf}


def calibrate(pilot_session_path):
    """Estimate mode/stage timing slopes from plan intervals and host2 timing rows.

    Reads only plan.json and host2/train.jsonl. Each point uses the five recorded
    fault steps and the preceding healthy interval after its first step.
    """
    session_root, plan_path, train_path = _session_paths(pilot_session_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    layout = plan.get("layout")
    _check_layout(layout)
    if plan.get("online") is not False or plan.get("probe_enabled") is not False \
            or plan.get("profile_enabled") is not False:
        raise ValueError("calibration requires an offline pilot with probe and PROFILE disabled")
    if plan.get("measurement_only") is not True:
        raise ValueError("calibration requires a measurement-only pilot plan")
    pilot_delay = plan.get("pilot_delay_ms_per_microbatch", PILOT_DELAY_MS_PER_MICROBATCH)
    if (isinstance(pilot_delay, bool) or not isinstance(pilot_delay, (int, float))
            or not math.isfinite(pilot_delay) or float(pilot_delay) not in (200.0, 400.0)):
        raise ValueError("pilot plan delay_ms_per_microbatch must be 200 or 400")
    pilot_delay = float(pilot_delay)
    intervals = plan.get("intervals")
    if not isinstance(intervals, list):
        raise ValueError("pilot plan has no interval list")
    by_id = {row.get("id"): row for row in intervals}
    if len(by_id) != len(intervals):
        raise ValueError("pilot interval ids must be unique")
    timing = _read_timing_rows(train_path)
    records = {}
    fault_intervals = [row for row in intervals if row.get("segment") == "fault"]
    if len(fault_intervals) != 8:
        raise ValueError(f"pilot must contain eight fault intervals, got {len(fault_intervals)}")

    for fault in fault_intervals:
        mode = fault.get("fault_mode")
        rank = fault.get("pilot_rank")
        pp_stage = fault.get("pp_stage")
        if mode not in MODES or rank not in (0, 15):
            raise ValueError(f"invalid pilot fault metadata: {fault.get('id')}")
        expected_stage = _pp_stage(layout, rank)
        if pp_stage != expected_stage or pp_stage != (0 if rank == 0 else 1):
            raise ValueError(f"pilot rank {rank} has inconsistent PP stage for {layout}")
        if fault.get("target_ranks") != [rank] or fault.get("delay_ms") != pilot_delay:
            raise ValueError(f"pilot injection metadata changed: {fault.get('id')}")
        key = f"{mode}:pp{pp_stage}"
        if key in records:
            raise ValueError(f"duplicate calibration point {key}")

        healthy = by_id.get(fault.get("baseline_interval_id"))
        if healthy is None or healthy.get("segment") != "healthy":
            raise ValueError(f"fault {fault.get('id')} has no preceding healthy interval")
        if healthy["end"] + 1 != fault["start"]:
            raise ValueError(f"fault {fault.get('id')} is not immediately preceded by its baseline")
        if fault.get("record_start") != fault["start"] + 1 \
                or healthy.get("record_start") != healthy["start"] + 1:
            raise ValueError("pilot record_start must omit the first iteration of each segment")

        healthy_first = max(int(healthy["record_start"]), int(healthy["start"]) + 1, 13)
        healthy_steps = list(range(healthy_first, int(healthy["end"]) + 1))
        fault_steps = list(range(int(fault["record_start"]), int(fault["end"]) + 1))
        if len(healthy_steps) != 7 or len(fault_steps) != 5:
            raise ValueError(f"pilot segment lengths changed for {fault.get('id')}")
        missing = [step for step in healthy_steps + fault_steps if step not in timing]
        if missing:
            raise ValueError(f"missing host2 timing rows for {fault.get('id')}: {missing}")
        healthy_values = [timing[step] for step in healthy_steps]
        fault_values = [timing[step] for step in fault_steps]
        healthy_stats = _sample_summary(healthy_values)
        fault_stats = _sample_summary(fault_values)
        healthy_mean, fault_mean = healthy_stats["mean_ms"], fault_stats["mean_ms"]
        delta = fault_mean - healthy_mean
        pooled_difference_se = math.sqrt(
            healthy_stats["sample_stddev_ms"] ** 2 / len(healthy_values)
            + fault_stats["sample_stddev_ms"] ** 2 / len(fault_values)
        )
        signal_to_noise = math.inf if pooled_difference_se == 0 else delta / pooled_difference_se
        noisy = (healthy_stats["coefficient_of_variation"] > MAX_SAMPLE_CV
                 or fault_stats["coefficient_of_variation"] > MAX_SAMPLE_CV
                 or signal_to_noise < MIN_SIGNAL_TO_NOISE)
        if delta <= 0:
            raise ValueError(f"non-positive fault timing increment for {key}: {delta:.6f} ms")
        if noisy:
            raise ValueError(
                f"pilot timing too noisy for {key}: healthy_cv="
                f"{healthy_stats['coefficient_of_variation']:.3f}, fault_cv="
                f"{fault_stats['coefficient_of_variation']:.3f}, signal_to_noise={signal_to_noise:.3f}"
            )
        slope = delta / pilot_delay
        records[key] = {
            "mode": mode,
            "pp_stage": pp_stage,
            "pilot_rank": rank,
            "pilot_interval_id": fault["id"],
            "baseline_interval_id": healthy["id"],
            "delay_ms_per_microbatch": pilot_delay,
            "healthy_iterations": healthy_steps,
            "fault_iterations": fault_steps,
            "healthy_wall_iteration_ms": healthy_values,
            "fault_wall_iteration_ms": fault_values,
            "healthy_mean_ms": healthy_mean,
            "fault_mean_ms": fault_mean,
            "fault_increment_ms": delta,
            "slope": slope,
            "slope_units": "wall_iteration_ms_per_delay_ms_per_microbatch",
            "healthy_sample_stddev_ms": healthy_stats["sample_stddev_ms"],
            "fault_sample_stddev_ms": fault_stats["sample_stddev_ms"],
            "healthy_coefficient_of_variation": healthy_stats["coefficient_of_variation"],
            "fault_coefficient_of_variation": fault_stats["coefficient_of_variation"],
            "difference_standard_error_ms": pooled_difference_se,
            "signal_to_noise": signal_to_noise,
        }

    expected_keys = {f"{mode}:pp{stage}" for mode in MODES for stage in (0, 1)}
    if set(records) != expected_keys:
        raise ValueError(f"calibration points missing or duplicated: {sorted(expected_keys - set(records))}")
    return {
        "schema_version": 1,
        "role": "mixed32_calibration",
        "session_id": plan.get("session_id"),
        "layout": layout,
        "delay_ms_per_microbatch": pilot_delay,
        "noise_gate": {
            "max_sample_coefficient_of_variation": MAX_SAMPLE_CV,
            "minimum_signal_to_noise": MIN_SIGNAL_TO_NOISE,
            "definition": "fault increment divided by pooled standard error; fixed before formal runs",
        },
        "measurement_source": "host2/train.jsonl wall_iteration_ms",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "by_mode_ppstage": records,
    }


def _get_calibration_record(calibration, mode, pp_stage, layout):
    if calibration.get("layout") != layout:
        raise ValueError(f"calibration layout {calibration.get('layout')!r} does not match {layout!r}")
    records = calibration.get("by_mode_ppstage")
    key = f"{mode}:pp{pp_stage}"
    record = records.get(key) if isinstance(records, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"missing calibration parameters for {key}")
    slope = record.get("slope")
    healthy_mean = record.get("healthy_mean_ms")
    if not isinstance(slope, (int, float)) or not math.isfinite(slope) or slope <= 0:
        raise ValueError(f"invalid calibration slope for {key}: {slope!r}")
    if not isinstance(healthy_mean, (int, float)) or not math.isfinite(healthy_mean) or healthy_mean <= 0:
        raise ValueError(f"invalid calibration healthy mean for {key}: {healthy_mean!r}")
    return record


def build_formal(session, layout, seed, design, calibration):
    """Build the 995-step online formal plan; this function never starts training."""
    _check_layout(layout)
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _validate_design(design)
    cases = list(design["cases"])
    random.Random(_case_order_seed(seed, layout)).shuffle(cases)

    intervals = []
    start = 1
    baseline = _make_interval("healthy_baseline", start, 45,
                              segment="healthy", baseline=True)
    intervals.append(baseline)
    start = baseline["end"] + 1
    previous_healthy_id = baseline["id"]
    formal_cases = []
    for index, case in enumerate(cases, 1):
        rank = int(case["rank"])
        mode = case["fault_mode"]
        pp_stage = _pp_stage(layout, rank)
        calibration_row = _get_calibration_record(calibration, mode, pp_stage, layout)
        healthy_mean = float(calibration_row["healthy_mean_ms"])
        slope = float(calibration_row["slope"])
        target = float(case["target_degradation_pct"])
        requested = healthy_mean * target / 100.0 / slope
        applied = min(FORMAL_DELAY_MAX_MS, max(FORMAL_DELAY_MIN_MS, requested))
        delay_ms = round(applied, 3)
        predicted = slope * delay_ms / healthy_mean * 100.0
        clipped = requested < FORMAL_DELAY_MIN_MS or requested > FORMAL_DELAY_MAX_MS
        if requested < FORMAL_DELAY_MIN_MS:
            clip_reason = "below_minimum"
        elif requested > FORMAL_DELAY_MAX_MS:
            clip_reason = "above_maximum"
        else:
            clip_reason = None

        case_id = f"formal{index:02d}_rank{rank:02d}_{mode}"
        fault = _make_interval(
            case_id, start, 25, segment="fault", case_id=case_id,
            target_ranks=[rank], delay_ms=delay_ms, fault_mode=mode,
            pp_stage=pp_stage, baseline_interval_id=previous_healthy_id,
            tier=int(case["tier"]), target_degradation_pct=target,
            tier_range_pct=list(case["tier_range_pct"]),
            dose_source="pilot_healthy_mean_ms * target_degradation_pct / 100 / calibrated_slope",
            dose_scope="per_microbatch total for the selected phase; split across hooked modules",
            delay_ms_per_microbatch=delay_ms,
            microbatches=MICROBATCHES,
            requested_total_ms_per_iteration=round(delay_ms * MICROBATCHES, 3),
            requested_total_fault_ms_per_rank=round(delay_ms * MICROBATCHES * 25, 3),
            unclamped_delay_ms=round(requested, 6),
            clamped_delay_ms=delay_ms,
            clip_min_ms=FORMAL_DELAY_MIN_MS,
            clip_max_ms=FORMAL_DELAY_MAX_MS,
            clipped=clipped,
            clip_reason=clip_reason,
            calibration_slope=slope,
            calibration_healthy_mean_ms=healthy_mean,
            predicted_degradation_pct=round(predicted, 6),
            prediction_error_pct_points=round(predicted - target, 6),
            design_case_id=case["case_id"],
            design_seed=design["seed"],
        )
        intervals.append(fault)
        start = fault["end"] + 1

        recovery_length = 35 if index < len(cases) else 25
        recovery_id = (f"healthy_after_formal{index:02d}" if index < len(cases)
                       else "healthy_final")
        recovery = _make_interval(
            recovery_id, start, recovery_length, segment="healthy",
            recovery_after_case=case_id, final=index == len(cases),
        )
        intervals.append(recovery)
        start = recovery["end"] + 1
        previous_healthy_id = recovery_id
        formal_cases.append({
            "case_id": case_id,
            "design_case_id": case["case_id"],
            "rank": rank,
            "fault_mode": mode,
            "pp_stage": pp_stage,
            "tier": int(case["tier"]),
            "target_degradation_pct": target,
            "delay_ms_per_microbatch": delay_ms,
            "clipped": clipped,
            "clip_reason": clip_reason,
            "predicted_degradation_pct": round(predicted, 6),
        })

    total_steps = start - 1
    if total_steps != 995:
        raise AssertionError(f"formal duration changed unexpectedly: {total_steps}")
    return {
        "schema_version": 1,
        "session_id": session,
        "role": "mixed32_formal",
        "layout": layout,
        "parallel_layout": layout,
        "seed": seed,
        "design_seed": design["seed"],
        "steps": total_steps,
        "online": True,
        "probe_enabled": True,
        "profile_enabled": False,
        "max_profiles": 24,
        "seq_len": 16,
        "microbatches": MICROBATCHES,
        "warmup_steps": 12,
        "fault_kind": "mixed_compute_simulation",
        "dose_definition": "delay_ms is the selected phase total for each microbatch; two microbatches inject two doses per iteration",
        "design_sha256": sha256_json(design),
        "calibration_sha256": sha256_json(calibration),
        "case_order": [case["case_id"] for case in formal_cases],
        "cases": formal_cases,
        "intervals": intervals,
    }


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def freeze_protocol_bundle(output_dir, design, pilots, calibrations, formal_plans):
    """Persist immutable JSON artifacts and their SHA-256 manifest before runs."""
    _validate_design(design)
    for name, value in (("pilots", pilots), ("calibrations", calibrations),
                        ("formal_plans", formal_plans)):
        if not isinstance(value, dict) or set(value) != set(LAYOUTS):
            raise ValueError(f"{name} must contain exactly {LAYOUTS}")
    artifacts = {"design.json": design}
    for layout in LAYOUTS:
        if pilots[layout].get("layout") != layout or calibrations[layout].get("layout") != layout \
                or formal_plans[layout].get("layout") != layout:
            raise ValueError(f"bundle contains a mismatched layout for {layout}")
        artifacts[f"pilot_{layout}.json"] = pilots[layout]
        artifacts[f"calibration_{layout}.json"] = calibrations[layout]
        artifacts[f"formal_{layout}.json"] = formal_plans[layout]
    payloads = {name: _json_bytes(value) for name, value in artifacts.items()}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    manifest = {
        "schema_version": 1,
        "protocol": "mixed32",
        "artifacts": hashes,
    }
    manifest_bytes = _json_bytes(manifest)
    payloads["freeze_manifest.json"] = manifest_bytes

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        destination = output / name
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace frozen artifact: {destination}")
    for name, payload in payloads.items():
        destination = output / name
        if not destination.exists():
            destination.write_bytes(payload)
    return {
        "directory": str(output),
        "artifacts": hashes,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
