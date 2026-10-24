"""CPU-only paired whole-compute protocol; never launches training or reads localization.

Freeze the design before pilots, then run one fixed-ratio pilot and one fitted-dose
verification per layout. Failures remain evidence: no automatic retries or tuning.
Sample lengths are explicit plan parameters, to be frozen before formal runs.
"""

import hashlib
import json
import math
from pathlib import Path
import random
import statistics

LAYOUTS = ("legacy", "node_local_pp")
DEFAULT_SEED = 20261023
MIN_VERIFIED_DEGRADATION_PCT = 30.0


def _number(value, name, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _length(value, name, minimum=2):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _layout(layout):
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_json(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def build_design(seed=DEFAULT_SEED):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    cases = [{"case_id": f"rank{rank:02d}_whole_compute", "rank": rank,
              "fault_mode": "whole_compute",
              "target_degradation_pct": round(rng.uniform(40, 80), 3)}
             for rank in range(16)]
    return {"schema_version": 1, "protocol": "whole32", "seed": seed,
            "pairing": "same_rank_and_target_across_layouts", "cases_per_layout": 16,
            "total_formal_cases": 32, "target_range_pct": [40, 80],
            "minimum_verified_degradation_pct": MIN_VERIFIED_DEGRADATION_PCT,
            "calibration_policy": "fixed_ratio_pilot_then_all_rank_fitted_dose_verification_no_automatic_retry",
            "formal_failure_policy": "retain_all_cases_including_low_strength_errors_and_false_alarms",
            "cases": cases}


def _design(design):
    cases = design.get("cases", [])
    if len(cases) != 16 or sorted(c.get("rank") for c in cases) != list(range(16)):
        raise ValueError("design must assign every rank exactly once")
    for case in cases:
        if case.get("fault_mode") != "whole_compute" or not 40 <= _number(case.get("target_degradation_pct"), "target") <= 80:
            raise ValueError("design requires whole_compute with targets in [40,80]")
    if design.get("minimum_verified_degradation_pct") != MIN_VERIFIED_DEGRADATION_PCT:
        raise ValueError("minimum verification gate changed")


def _order(seed, layout):
    ranks = list(range(16))
    random.Random(seed ^ (0x9E3779B1 if layout == "legacy" else 0x85EBCA77)).shuffle(ranks)
    return ranks


def _interval(identifier, start, length, **fields):
    row = {"id": identifier, "start": start, "end": start + length - 1,
           "record_start": start + 1, "target_ranks": [], "fault_mode": None,
           "delay_ms": 0.0, "slowdown_extra_ratio": 0.0}
    row.update(fields)
    return row


def _base(session, layout, seed, role, intervals, **fields):
    _layout(layout)
    plan = {"schema_version": 1, "protocol": "whole32", "session_id": session,
            "role": role, "layout": layout, "parallel_layout": layout, "seed": seed,
            "steps": intervals[-1]["end"], "seq_len": 16, "microbatches": 2,
            "warmup_steps": 12, "fault_kind": "whole_compute_simulation",
            "online": False, "probe_enabled": False, "profile_enabled": False,
            "max_profiles": 0, "measurement_only": True,
            "measurement_source": "host2/train.jsonl wall_iteration_ms only",
            "dose_definition": "each major compute op adds FMA proportional to its healthy CUDA duration; multiplier=1+slowdown_extra_ratio",
            "calibration_artifact_glob": "**/whole_compute_calibration_rank*.json",
            "retain_calibration_artifacts": True, "intervals": intervals}
    plan.update(fields)
    return plan


def _offline(session, layout, seed, role, ratios, healthy_steps, fault_steps, opening_steps, ranks=None, **fields):
    _layout(layout)
    _length(healthy_steps, "healthy_steps", 3)
    _length(fault_steps, "fault_steps", 3)
    _length(opening_steps, "opening_steps", 13)
    intervals = [_interval("healthy_opening", 1, opening_steps, segment="healthy")]
    start = opening_steps + 1
    selected = list(range(16)) if ranks is None else list(ranks)
    if not selected or len(set(selected)) != len(selected) or any(r not in range(16) for r in selected):
        raise ValueError("invalid calibration ranks")
    for index, rank in enumerate([r for r in _order(seed, layout) if r in selected], 1):
        healthy = _interval(f"healthy_before_rank{rank:02d}", start, healthy_steps, segment="healthy")
        healthy["record_start"] = healthy["start"] + 2
        intervals.append(healthy)
        ratio = _number(ratios[str(rank)], f"rank{rank} ratio")
        fault = _interval(f"{role}_rank{rank:02d}", healthy["end"] + 1, fault_steps,
                          segment="fault", case_index=index, rank=rank,
                          target_ranks=[rank], fault_mode="whole_compute", delay_ms=1.0,
                          slowdown_extra_ratio=ratio, baseline_interval_id=healthy["id"])
        fault["record_start"] = fault["start"] + 2
        intervals.append(fault)
        start = fault["end"] + 1
    intervals.append(_interval("healthy_final", start, healthy_steps, segment="healthy"))
    return _base(session, layout, seed, role, intervals, healthy_steps=healthy_steps,
                 fault_steps=fault_steps, opening_steps=opening_steps, calibration_ranks=selected, **fields)


def build_pilot(session, layout, seed=DEFAULT_SEED, ratio=8.0, healthy_steps=8, fault_steps=8, opening_steps=24):
    """All 16 ranks receive the same extra compute ratio, one at a time."""
    ratio = _number(ratio, "ratio")
    return _offline(session, layout, seed, "whole32_pilot", {str(r): ratio for r in range(16)},
                    healthy_steps, fault_steps, opening_steps, pilot_ratio=ratio)


def _measure(session_path, expected_role):
    root = Path(session_path)
    if root.name == "train.jsonl":
        root = root.parent.parent
    plan_path, train_path = root / "plan.json", root / "host2" / "train.jsonl"
    plan = json.loads(plan_path.read_text())
    _layout(plan.get("layout"))
    if plan.get("role") != expected_role or any(plan.get(key) is not False for key in ("online", "probe_enabled", "profile_enabled")) or plan.get("measurement_only") is not True:
        raise ValueError("timing calibration requires the expected offline measurement-only plan")
    timing = {}
    for line in train_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("iteration", row.get("step")))
        if step in timing:
            raise ValueError(f"duplicate wall timing iteration {step}")
        timing[step] = _number(row["wall_iteration_ms"], "wall_iteration_ms")
    by_id = {i["id"]: i for i in plan["intervals"]}
    records = {}
    for fault in plan["intervals"]:
        if fault.get("segment") != "fault":
            continue
        ranks = fault.get("target_ranks", [])
        if len(ranks) != 1 or ranks[0] not in range(16) or fault.get("fault_mode") != "whole_compute":
            raise ValueError("offline faults must target one valid rank using whole_compute")
        rank = ranks[0]
        if str(rank) in records:
            raise ValueError(f"duplicate rank {rank}")
        healthy = by_id[fault["baseline_interval_id"]]
        if healthy.get("segment") != "healthy" or healthy["end"] + 1 != fault["start"]:
            raise ValueError("each fault must follow its healthy reference")
        samples = {}
        for name, segment in (("healthy", healthy), ("fault", fault)):
            first = max(segment["start"] + 1, segment["record_start"])
            steps = list(range(first, segment["end"] + 1))
            if len(steps) < 1:
                raise ValueError("empty timing sample")
            values = [timing[step] for step in steps]
            mean = statistics.fmean(values)
            samples[name + "_iterations"] = steps
            samples[name + "_wall_iteration_ms"] = values
            samples[name + "_mean_ms"] = mean
            samples[name + "_cv"] = statistics.stdev(values) / mean if len(values) > 1 else 0.0
        degradation = 100 * (samples["fault_mean_ms"] / samples["healthy_mean_ms"] - 1)
        records[str(rank)] = {"rank": rank, "slowdown_extra_ratio": _number(fault["slowdown_extra_ratio"], "ratio"),
                              "actual_degradation_pct": degradation, **samples}
    if set(records) != {str(r) for r in plan.get("calibration_ranks", range(16))}:
        raise ValueError("all planned rank measurements are required")
    artifacts = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.glob("**/whole_compute_calibration_rank*.json"))}
    return {"schema_version": 1, "protocol": "whole32", "layout": plan["layout"],
            "session_id": plan["session_id"], "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "measurement_source": "host2/train.jsonl wall_iteration_ms only",
            "calibration_artifacts_sha256": artifacts, "by_rank": records}, plan


def calibrate(pilot_session_path):
    """Return every raw point, including unusable or low-strength points."""
    result, _ = _measure(pilot_session_path, "whole32_pilot")
    for row in result["by_rank"].values():
        row["slope_pct_per_extra_ratio"] = row["actual_degradation_pct"] / row["slowdown_extra_ratio"]
        row["fit_usable"] = row["slope_pct_per_extra_ratio"] > 0
    result["role"] = "whole32_calibration"
    return result


def build_validation(session, layout, seed, design, calibration, healthy_steps=8, fault_steps=8, opening_steps=24, ratio_overrides=None, ranks=None):
    """Fit each rank independently; any explicit correction is recorded and reverified."""
    _design(design)
    if calibration.get("layout") != layout:
        raise ValueError("calibration layout mismatch")
    ratios = {}
    for case in design["cases"]:
        key = str(case["rank"])
        slope = _number(calibration["by_rank"][key]["slope_pct_per_extra_ratio"], "slope")
        ratios[key] = round(case["target_degradation_pct"] / slope, 6)
    overrides = {str(k): _number(v, "override ratio") for k, v in (ratio_overrides or {}).items()}
    if set(overrides) - set(ratios):
        raise ValueError("invalid override ranks")
    ratios.update(overrides)
    return _offline(session, layout, seed, "whole32_validation", ratios, healthy_steps, fault_steps,
                    opening_steps, ranks=ranks, design_sha256=sha256_json(design),
                    calibration_sha256=sha256_json(calibration), explicit_ratio_overrides=overrides,
                    whole_compute_reference_session=calibration["session_id"])


def validate_strength(validation_session_path, design, previous_validation=None):
    _design(design)
    result, plan = _measure(validation_session_path, "whole32_validation")
    if plan.get("design_sha256") != sha256_json(design):
        raise ValueError("validation design hash mismatch")
    result.update(role="whole32_strength_validation", design_sha256=sha256_json(design),
                  calibration_sha256=plan["calibration_sha256"],
                  whole_compute_reference_session=plan["whole_compute_reference_session"],
                  minimum_degradation_pct=MIN_VERIFIED_DEGRADATION_PCT)
    if previous_validation is not None:
        if (previous_validation.get("layout") != result["layout"]
                or previous_validation.get("design_sha256") != sha256_json(design)
                or previous_validation.get("whole_compute_reference_session") != result["whole_compute_reference_session"]):
            raise ValueError("previous verification layout/design mismatch")
        if previous_validation.get("correction_round", 0) != 0:
            raise ValueError("at most one explicit correction round is allowed")
        result["previous_validation"] = previous_validation
        result["previous_validation_sha256"] = sha256_json(previous_validation)
        result["correction_round"] = 1
        result["corrected_ranks"] = sorted(int(r) for r in result["by_rank"])
        result["by_rank"] = {**previous_validation["by_rank"], **result["by_rank"]}
    else:
        result["correction_round"] = 0
    if set(result["by_rank"]) != {str(r) for r in range(16)}:
        raise ValueError("initial verification requires every rank")
    for case in design["cases"]:
        row = result["by_rank"][str(case["rank"])]
        row.update(target_degradation_pct=case["target_degradation_pct"],
                   strength_passed=row["actual_degradation_pct"] >= MIN_VERIFIED_DEGRADATION_PCT)
    result["all_ranks_passed"] = all(r["strength_passed"] for r in result["by_rank"].values())
    return result


def build_formal(session, layout, seed, design, validation, opening_steps=60, recovery_steps=35, fault_steps=25):
    """Use only verified doses, retaining every online event and all 32 formal cases."""
    _design(design)
    _length(opening_steps, "opening_steps", 60)
    _length(recovery_steps, "recovery_steps", 25)
    if fault_steps != 25:
        raise ValueError("formal fault duration is fixed at 25 iterations")
    if validation.get("layout") != layout or validation.get("design_sha256") != sha256_json(design):
        raise ValueError("validation layout/design mismatch")
    if not validation.get("all_ranks_passed") or set(validation.get("by_rank", {})) != {str(r) for r in range(16)}:
        raise ValueError("formal plan requires measured strength confirmation for every rank")
    intervals = [_interval("healthy_baseline", 1, opening_steps, segment="healthy", baseline=True)]
    cases = {c["rank"]: c for c in design["cases"]}
    for index, rank in enumerate(_order(seed, layout), 1):
        measured = validation["by_rank"][str(rank)]
        if measured["actual_degradation_pct"] < MIN_VERIFIED_DEGRADATION_PCT:
            raise ValueError(f"rank {rank} does not meet actual strength gate")
        case_id = f"formal{index:02d}_rank{rank:02d}_whole_compute"
        fault = _interval(case_id, intervals[-1]["end"] + 1, fault_steps, segment="fault", case_id=case_id,
                          design_case_id=cases[rank]["case_id"], target_ranks=[rank], rank=rank,
                          fault_mode="whole_compute", delay_ms=1.0,
                          slowdown_extra_ratio=_number(measured["slowdown_extra_ratio"], "ratio"),
                          target_degradation_pct=cases[rank]["target_degradation_pct"],
                          verified_degradation_pct=measured["actual_degradation_pct"],
                          baseline_interval_id=intervals[-1]["id"])
        intervals.append(fault)
        intervals.append(_interval(f"healthy_after_formal{index:02d}", fault["end"] + 1,
                                   recovery_steps, segment="healthy", recovery_after_case=case_id))
    return _base(session, layout, seed, "whole32_formal", intervals,
                 online=True, probe_enabled=True, measurement_only=False, max_profiles=24,
                 opening_steps=opening_steps, recovery_steps=recovery_steps, fault_steps=25,
                 design_sha256=sha256_json(design), validation_sha256=sha256_json(validation),
                 whole_compute_reference_session=validation["whole_compute_reference_session"],
                 cases=[dict(i) for i in intervals if i.get("segment") == "fault"],
                 retain_all_false_alarms=True, detector_and_localizer_frozen=True)


def freeze_protocol_bundle(output_dir, design, pilots, calibrations, validations, formal_plans):
    """Write immutable inputs before formal execution; refuse any replacement."""
    _design(design)
    artifacts = {"design.json": design}
    for label, collection in (("pilot", pilots), ("calibration", calibrations),
                              ("validation", validations), ("formal", formal_plans)):
        if set(collection) != set(LAYOUTS):
            raise ValueError(f"{label} requires both layouts")
        for layout, value in collection.items():
            if value.get("layout") != layout:
                raise ValueError(f"{label} layout mismatch")
            artifacts[f"{label}_{layout}.json"] = value
    for layout in LAYOUTS:
        formal, verified = formal_plans[layout], validations[layout]
        reference = calibrations[layout].get("session_id")
        if (not reference or reference != pilots[layout].get("session_id")
                or verified.get("whole_compute_reference_session") != reference
                or formal.get("whole_compute_reference_session") != reference):
            raise ValueError("verification/formal must reuse their layout pilot compute calibration")
        if not verified.get("all_ranks_passed") or formal.get("design_sha256") != sha256_json(design) or formal.get("validation_sha256") != sha256_json(verified):
            raise ValueError("formal plans must reference passed verification and frozen design")
    payloads = {name: _json_bytes(value) for name, value in artifacts.items()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    payloads["freeze_manifest.json"] = _json_bytes({"schema_version": 1, "protocol": "whole32", "artifacts": hashes})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    for name, data in payloads.items():
        with (output / name).open("ab") as stream:
            if stream.tell() == 0:
                stream.write(data)
    return {"directory": str(output), "artifacts": hashes,
            "manifest_sha256": hashlib.sha256(payloads["freeze_manifest.json"]).hexdigest()}
