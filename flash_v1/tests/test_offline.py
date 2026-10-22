"""Synthetic logic tests only; these are not GPU experiment results."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flash_v1.offline import (
    FeatureSelectionUnavailable,
    _run_diagnostics,
    evaluate_dataset,
    extract_run_features,
    extract_run_features_data,
    fit_dataset,
)


TOPOLOGY = {
    "schema_version": 2,
    "world_size": 3,
    "groups": [
        {"name": "dp_01", "label": "DP01", "localization": True,
         "members": [0, 1], "reporters": [0, 1], "kind": "all_reduce"},
        {"name": "dp_12", "label": "DP12", "localization": True,
         "members": [1, 2], "reporters": [1, 2], "kind": "all_reduce"},
        {"name": "dynamic_ep", "label": "EP", "localization": False,
         "members": [0, 1, 2], "reporters": [0, 1, 2], "kind": "all_to_all"},
    ],
}

TP_TOPOLOGY = {
    "schema_version": 2,
    "world_size": 2,
    "groups": [
        {"name": "tensor_parallel", "label": "TP_PP0", "localization": True,
         "members": [0, 1], "reporters": [0, 1], "kind": "all_reduce"},
        {"name": "dp_01", "label": "data_parallel", "localization": True,
         "members": [0, 1], "reporters": [0, 1], "kind": "all_reduce"},
    ],
}


def _profile(fault_rank, ratio=0.5):
    rows = []
    groups = [("DP01", "dp_01", [0, 1]), ("DP12", "dp_12", [1, 2])]
    for iteration in range(4):
        for label, name, members in groups:
            for rank in members:
                duration = 100.0
                if fault_rank == rank:
                    duration *= ratio
                rows.append({
                    "rank": rank,
                    "clique": f"volatile-clique-{iteration}-{rank}",
                    "group_name": name,
                    "tag": label,
                    "group_members": members,
                    "kind": "all_reduce",
                    "count": 4,
                    "dtype": 9,
                    "duration_us": duration,
                    "iteration": iteration,
                    "group_id": 9000 + iteration * 10 + rank,
                    "timestamp_us": 1_000_000 + iteration * 1000 + rank,
                })
    # Dynamic records are not marked stable in topology and must not enter features.
    rows.append({
        "rank": 0, "clique": "dynamic", "group_name": "dynamic_ep", "tag": "EP",
        "group_members": [0, 1, 2], "kind": "all_to_all", "count": 128,
        "dtype": 9, "duration_us": 1_000_000.0, "iteration": 0,
        "group_id": 1, "timestamp_us": 2,
    })
    return rows


def _with_phase_records(profile):
    expanded = []
    for row in profile:
        if row["group_name"] not in {"dp_01", "dp_12"}:
            expanded.append(row)
            continue
        for phase_code, phase in ((257, "attn_fwd"), (258, "attn_bwd")):
            phased = dict(row, phase_code=phase_code, layer=0, phase=phase)
            expanded.append(phased)
    return expanded


def _write_run(root, run_id, session_id, role, fault_rank, ratio=0.5, include_phases=False):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "session_id": session_id,
        "role": role,
        "fault_rank": fault_rank,
        "config": {"model": "synthetic-test"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "topology.json").write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    profile = _profile(fault_rank, ratio)
    if include_phases:
        profile = _with_phase_records(profile)
    (run_dir / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return run_dir


def _dataset(root, include_phases=False):
    for suffix, role, ratios in (
        ("train-a", "train", {None: 1.0, 0: 0.50, 1: 0.50, 2: 0.50}),
        ("train-b", "train", {None: 1.0, 0: 0.54, 1: 0.54, 2: 0.54}),
        ("validation", "validation", {None: 1.0, 0: 0.52, 1: 0.52, 2: 0.52}),
        ("blind", "blind_test", {None: 1.0, 0: 0.53, 1: 0.53, 2: 0.53}),
    ):
        for fault_rank, ratio in ratios.items():
            run_id = f"{suffix}-healthy" if fault_rank is None else f"{suffix}-rank{fault_rank}"
            _write_run(root, run_id, suffix, role, fault_rank, ratio, include_phases)


def _phase_call_profile(fault_rank=None):
    rows = []
    for block, start in enumerate((441, 885)):
        for offset in range(3):
            operation_index = start + offset
            for rank in (0, 1):
                if offset == 0 and fault_rank is not None:
                    duration = 10.0 if rank == fault_rank else 16_520.0
                elif offset == 0:
                    duration = 1_000.0
                else:
                    duration = 10.0 + offset
                rows.append({
                    "rank": rank,
                    "clique": f"changes-{block}-{rank}",
                    "group_name": "tensor_parallel",
                    "tag": "TP_PP0",
                    "group_members": [0, 1],
                    "kind": "all_reduce",
                    "count": 64,
                    "dtype": 9,
                    "phase_code": 258,
                    "layer": 0,
                    "phase": "attention_backward",
                    "operation_index": operation_index,
                    "duration_us": duration,
                    "iteration": block,
                    "group_id": 100 * block + rank,
                    "timestamp_us": 1_000_000 + operation_index + rank,
                })
    # Stable non-TP rows are deliberately present but are outside this selector's scope.
    for row in list(rows):
        rows.append(dict(row, group_name="dp_01", tag="data_parallel", group_members=[0, 1]))
    return rows


def _call_dataset(root):
    for suffix, role, labels in (
        ("train-a", "train", (None, 0, 1)),
        ("train-b", "train", (None, 0, 1)),
        ("validation", "validation", (None, 0, 1)),
        ("blind", "blind_test", (None, 0, 1)),
    ):
        for fault_rank in labels:
            run_id = f"{suffix}-healthy" if fault_rank is None else f"{suffix}-rank{fault_rank}"
            run_dir = root / run_id
            run_dir.mkdir(parents=True)
            manifest = {
                "run_id": run_id,
                "session_id": suffix,
                "role": role,
                "fault_rank": fault_rank,
                "configuration": {
                    "model": "Qwen-synthetic-test",
                    "attention_backward_block_size": 3,
                    "microbatches": 2,
                    "sample_stride": 8,
                },
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "topology.json").write_text(json.dumps(TP_TOPOLOGY), encoding="utf-8")
            (run_dir / "profile.json").write_text(
                json.dumps(_phase_call_profile(fault_rank)), encoding="utf-8"
            )


class OfflineFeatureTests(unittest.TestCase):
    def test_features_ignore_ep_and_volatile_ids(self):
        profile = _profile(0)
        expected = extract_run_features_data(TOPOLOGY, profile)
        changed = json.loads(json.dumps(profile))
        for index, row in enumerate(changed):
            row["clique"] = f"different-{index}"
            row["group_id"] = -index
            row["timestamp_us"] = 99_000_000 + index
        actual = extract_run_features_data(TOPOLOGY, changed)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 4)
        self.assertTrue(all("EP" not in key for key in actual))

    def test_phase_code_is_identity_and_attention_backward_filter_uses_code(self):
        profile = _with_phase_records(_profile(0))
        all_features = extract_run_features_data(TOPOLOGY, profile, phases="all")
        backward_features = extract_run_features_data(TOPOLOGY, profile, phases="attention_backward")
        all_codes = {json.loads(feature_id)["phase_code"] for feature_id in all_features}
        backward_codes = {json.loads(feature_id)["phase_code"] for feature_id in backward_features}
        self.assertEqual(all_codes, {257, 258})
        self.assertEqual(backward_codes, {258})
        self.assertEqual(len(all_features), 8)
        self.assertEqual(len(backward_features), 4)
        derived = [dict(row) for row in profile]
        for row in derived:
            row.pop("phase_code", None)
        derived_features = extract_run_features_data(TOPOLOGY, derived)
        self.assertEqual({json.loads(key)["phase_code"] for key in derived_features}, {257, 258})

    def test_fit_uses_multiple_train_sessions_and_excludes_blind_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _dataset(root)
            model = fit_dataset(root)
            self.assertEqual(model["config"]["phases"], "all")
            roles = {run["role"] for run in model["source_runs"]}
            self.assertEqual(roles, {"train", "validation"})
            self.assertEqual(model["training_counts"]["train_sessions"], ["train-a", "train-b"])
            self.assertEqual(model["held_out_sessions"], ["blind"])
            validation_report = evaluate_dataset(root, model, split="validation")
            self.assertEqual(validation_report["run_denominator"], 4)
            self.assertEqual(validation_report["run_accuracy"], 1.0)
            report = evaluate_dataset(root, model, split="blind_test")
            self.assertEqual(report["run_denominator"], 4)
            self.assertEqual(report["run_accuracy"], 1.0)
            self.assertEqual(report["inconclusive_runs"], 0)
            self.assertEqual(report["healthy_false_positives"], 0)
            self.assertEqual(report["healthy_false_positive_denominator"], 1)
            self.assertEqual(report["confusion_matrix"]["rank:1"]["rank:1"], 1)

    def test_phase_filter_is_frozen_and_applied_during_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _dataset(root, include_phases=True)
            model = fit_dataset(root, phases="attention_backward")
            self.assertEqual(model["config"]["phases"], "attention_backward")
            self.assertEqual({definition["phase_code"] for definition in model["feature_definitions"]}, {258})
            with mock.patch("flash_v1.offline.extract_run_features", wraps=extract_run_features) as extract:
                report = evaluate_dataset(root, model, split="blind_test")
            self.assertTrue(extract.called)
            self.assertTrue(all(call.kwargs["phases"] == "attention_backward" for call in extract.call_args_list))
            self.assertEqual(report["run_denominator"], 4)

    def test_first_in_phase_preserves_wait_signal_without_later_calls_diluting_it(self):
        profile = _phase_call_profile(fault_rank=0)
        all_calls = extract_run_features_data(
            TP_TOPOLOGY, profile, phases="attention_backward", call_selection="all"
        )
        first_calls = extract_run_features_data(
            TP_TOPOLOGY, profile, phases="attention_backward", call_selection="first_in_phase"
        )
        definitions = {json.loads(key)["rank"]: key for key in first_calls}
        rank0 = definitions[0]
        self.assertGreater(first_calls[rank0], 7.0)
        self.assertLess(abs(all_calls[rank0]), 0.1)
        self.assertTrue(all(json.loads(key)["group_name"] == "tensor_parallel" for key in first_calls))

    def test_first_in_phase_fails_closed_when_segments_are_incomplete(self):
        profile = _phase_call_profile(fault_rank=0)
        # Removing one shared index creates segments whose lengths no longer
        # match the frozen two-microbatch, three-call phase structure.
        incomplete = [row for row in profile if row["operation_index"] != 442]
        with self.assertRaisesRegex(FeatureSelectionUnavailable, "coverage is incomplete"):
            extract_run_features_data(
                TP_TOPOLOGY,
                incomplete,
                phases="attention_backward",
                call_selection="first_in_phase",
            )
        with self.assertRaisesRegex(ValueError, "requires phases"):
            extract_run_features_data(
                TP_TOPOLOGY, profile, phases="all", call_selection="first_in_phase"
            )

    def test_first_in_phase_requires_phase_operation_and_member_coverage(self):
        profile = _phase_call_profile(fault_rank=0)
        missing_operation = [dict(row) for row in profile]
        missing_operation[0].pop("operation_index")
        with self.assertRaisesRegex(FeatureSelectionUnavailable, "operation_index is required"):
            extract_run_features_data(
                TP_TOPOLOGY, missing_operation, phases="attention_backward",
                call_selection="first_in_phase",
            )

        missing_phase = [dict(row) for row in profile]
        missing_phase[0].pop("phase_code")
        missing_phase[0].pop("phase")
        missing_phase[0].pop("layer")
        with self.assertRaisesRegex(FeatureSelectionUnavailable, "phase metadata is required"):
            extract_run_features_data(
                TP_TOPOLOGY, missing_phase, phases="attention_backward",
                call_selection="first_in_phase",
            )

        missing_member = [
            row for row in profile
            if not (row["group_name"] == "tensor_parallel" and row["operation_index"] == 442 and row["rank"] == 0)
        ]
        with self.assertRaisesRegex(FeatureSelectionUnavailable, "missing tensor_parallel members"):
            extract_run_features_data(
                TP_TOPOLOGY, missing_member, phases="attention_backward",
                call_selection="first_in_phase",
            )

    def test_first_in_phase_model_freezes_selector_and_labels_never_enter_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _call_dataset(root)
            model = fit_dataset(
                root, phases="attention_backward", call_selection="first_in_phase"
            )
            self.assertEqual(model["config"]["call_selection"], "first_in_phase")
            self.assertEqual(model["config"]["expected_phase_block_size"], 3)
            self.assertEqual(model["config"]["expected_blocks"], 2)
            self.assertEqual(model["config"]["sampling_stride"], 8)
            self.assertIn("Qwen TP2 PP2", model["feature_spec"]["call_selection_assumptions"]["scope"])
            blind_profile = root / "blind-rank0" / "profile.json"
            blind_topology = root / "blind-rank0" / "topology.json"
            before = extract_run_features(
                blind_topology, blind_profile,
                phases="attention_backward", call_selection="first_in_phase",
            )
            manifest_path = root / "blind-rank0" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original_label = manifest["fault_rank"]
            manifest["fault_rank"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            after = extract_run_features(
                blind_topology, blind_profile,
                phases="attention_backward", call_selection="first_in_phase",
            )
            self.assertEqual(before, after)
            manifest["fault_rank"] = original_label
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch("flash_v1.offline.extract_run_features", wraps=extract_run_features) as extract:
                report = evaluate_dataset(root, model, split="blind_test")
            self.assertTrue(extract.called)
            self.assertTrue(all(
                call.kwargs["call_selection"] == "first_in_phase" and
                call.kwargs["phases"] == "attention_backward"
                for call in extract.call_args_list
            ))
            self.assertEqual(report["run_denominator"], 3)
            self.assertEqual(report["run_accuracy"], 1.0)

    def test_missing_coverage_is_explicit_and_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _dataset(root)
            model = fit_dataset(root)
            run_dir = root / "blind-rank0"
            rows = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
            rows = [row for row in rows if row["group_name"] != "dp_12"]
            (run_dir / "profile.json").write_text(json.dumps(rows), encoding="utf-8")
            report = evaluate_dataset(root, model, split="blind_test")
            prediction = next(row for row in report["runs"] if row["run_id"] == "blind-rank0")
            self.assertEqual(prediction["reason"], "insufficient_feature_coverage")
            self.assertEqual(prediction["observed_feature_count"], 2)
            self.assertEqual(prediction["missing_feature_count"], 2)
            self.assertEqual(prediction["predicted"], "inconclusive")

    def test_midpoint_between_fault_templates_is_refused_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _dataset(root)
            model = fit_dataset(root)
            rank0 = model["templates"]["0"][0]["values"]
            rank1 = model["templates"]["1"][0]["values"]
            midpoint = {key: (rank0[key] + rank1[key]) / 2 for key in model["feature_ids"]}
            raw = {
                key: model["healthy_baselines"][key]["center"] +
                     model["healthy_baselines"][key]["scale"] * midpoint[key]
                for key in midpoint
            }
            details = _run_diagnostics({"run_id": "synthetic-midpoint", "session_id": "x", "features": raw}, model)
            self.assertEqual(details["predicted_label"], "inconclusive")
            self.assertEqual(details["reason"], "classes_too_close")
            self.assertLess(details["class_margin"], model["thresholds"]["min_class_margin"])

    def test_empty_observation_is_valid_json_and_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _dataset(root)
            model = fit_dataset(root)
            details = _run_diagnostics(
                {"run_id": "empty", "session_id": "empty-session", "features": {}}, model
            )
            self.assertEqual(details["predicted_label"], "inconclusive")
            self.assertEqual(details["reason"], "insufficient_feature_coverage")
            json.dumps(details, allow_nan=False)

    def test_fit_rejects_a_session_split_across_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data16"
            root.mkdir()
            _dataset(root)
            manifest_path = root / "blind-rank0" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["session_id"] = "train-a"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "session split leakage"):
                fit_dataset(root)


if __name__ == "__main__":
    unittest.main()
