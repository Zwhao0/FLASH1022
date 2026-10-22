"""Synthetic logic tests only; no GPU measurements are asserted."""

import json
import tempfile
import unittest
from pathlib import Path

from flash_v1.wait_localize import (
    CoverageError,
    evaluate_dataset,
    fit_dataset,
    localize_scores,
    score,
)


def _topology():
    groups = []
    for low in range(0, 16, 2):
        stage = low // 8
        groups.append({
            "name": "tensor_parallel",
            "label": f"TP_PP{stage}",
            "localization": True,
            "members": [low, low + 1],
            "reporters": [low, low + 1],
            "kind": 6,
        })
    return {"world_size": 16, "groups": groups}


def _profile(fault_rank=None):
    rows = []
    for low in range(0, 16, 2):
        stage = low // 8
        members = [low, low + 1]
        for local_layer in range(12):
            layer = stage * 12 + local_layer
            phase_code = ((layer + 1) << 8) | 2
            start = 100 + layer * 1000
            for block, first in enumerate((start, start + 10)):
                iteration = layer * 2 + block
                for offset in range(3):
                    operation_index = first + offset
                    for rank in members:
                        duration = 10.0 + rank * 0.1
                        if fault_rank in members and offset == 0:
                            if rank == fault_rank:
                                duration = 10.0
                            else:
                                duration = 20_010.0
                        rows.append({
                            "rank": rank,
                            "tag": f"TP_PP{stage}",
                            "group_name": "tensor_parallel",
                            "group_members": members,
                            "kind": 6,
                            "count": 14336,
                            "dtype": 9,
                            "phase_code": phase_code,
                            "operation_index": operation_index,
                            "iteration": iteration,
                            "duration_us": duration,
                            "clique": f"volatile-{rank}-{iteration}",
                            "group_id": 10_000 + iteration,
                            "timestamp_us": 99_000_000 + iteration,
                        })
    return rows


def _write_run(root, run_id, role, session, fault_rank=None):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "session_id": session,
        "role": role,
        "fault_rank": fault_rank,
        "configuration": {
            "model_id": "synthetic-24-layer-Qwen",
            "tp": 2,
            "pp": 2,
            "world_size": 16,
            "microbatches": 2,
            "attention_backward_block_size": 3,
            "sample_stride": 8,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "topology.json").write_text(json.dumps(_topology()), encoding="utf-8")
    (run_dir / "profile.json").write_text(json.dumps(_profile(fault_rank)), encoding="utf-8")
    return run_dir


def _dataset(root):
    _write_run(root, "train-s1-healthy", "train", "train-s1", None)
    _write_run(root, "train-s2-healthy", "train", "train-s2", None)
    _write_run(root, "train-s1-fault", "train", "train-s1", 3)
    _write_run(root, "validation-healthy", "validation", "validation-s3", None)
    _write_run(root, "validation-fault", "validation", "validation-s3", 7)


class WaitLocalizationTests(unittest.TestCase):
    def test_real_pair_score_is_signed_and_aggregated_over_12_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _write_run(Path(directory), "fault-rank3", "train", "s1", 3)
            result = score(run)
            self.assertEqual(result.coverage["observed_ranks"], 16)
            self.assertEqual(result.coverage["observed_rank_phases"], 16 * 12)
            self.assertEqual(result.coverage["observed_microbatch_pairs"], 16 * 12 * 2)
            self.assertEqual(len(result.phase_scores_us[3]), 12)
            self.assertGreater(result.rank_scores_us[3], 19_000)
            self.assertLess(result.rank_scores_us[2], -19_000)
            self.assertEqual(max(result.rank_scores_us, key=result.rank_scores_us.get), 3)

    def test_truth_and_later_calls_do_not_change_first_call_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _write_run(Path(directory), "rank3", "train", "s1", 3)
            before = score(run).rank_scores_us
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["fault_rank"] = 14
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            after_label_edit = score(run).rank_scores_us
            self.assertEqual(before, after_label_edit)

            profile_path = run / "profile.json"
            rows = json.loads(profile_path.read_text(encoding="utf-8"))
            for row in rows:
                if row["operation_index"] % 1000 not in (100, 110):
                    row["duration_us"] *= 100
            profile_path.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(before, score(run).rank_scores_us)

    def test_missing_member_pair_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _write_run(Path(directory), "fault-rank3", "train", "s1", 3)
            profile_path = run / "profile.json"
            rows = json.loads(profile_path.read_text(encoding="utf-8"))
            rows = [
                row for row in rows
                if not (
                    row["rank"] == 3 and row["phase_code"] == 258 and
                    row["operation_index"] == 100
                )
            ]
            profile_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises(CoverageError):
                score(run)

    def test_healthy_only_fit_freezes_thresholds_and_ignores_fault_rank_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            _dataset(root)
            model = fit_dataset(root)
            self.assertEqual(model["thresholds"]["healthy_runs_used"], 3)
            self.assertEqual(len(model["source_runs"]), 3)
            self.assertTrue(all("fault_rank" not in run for run in model["source_runs"]))
            self.assertGreater(model["thresholds"]["scale_h_us"], 0)
            validation_path = root / "validation-fault" / "manifest.json"
            manifest = json.loads(validation_path.read_text(encoding="utf-8"))
            manifest["fault_rank"] = 12
            validation_path.write_text(json.dumps(manifest), encoding="utf-8")
            changed = fit_dataset(root)
            self.assertEqual(model["thresholds"], changed["thresholds"])
            self.assertEqual(model["frozen_model_hash"], changed["frozen_model_hash"])

            report = evaluate_dataset(root, model, split="validation")
            fault = next(run for run in report["runs"] if run["run_id"] == "validation-fault")
            self.assertEqual(fault["predicted"], "rank:7")
            self.assertEqual(fault["true_label"], "rank:12")
            self.assertEqual(report["run_denominator"], 2)
            self.assertEqual(report["healthy_false_positive_denominator"], 1)

    def test_hysteresis_and_ratio_gate_are_explicit(self):
        thresholds = {
            "healthy_limit_us": 1.25,
            "fault_limit_us": 2.0,
            "minimum_top_to_runner_up_ratio": 2.0,
            "ratio_denominator_floor_us": 1.0,
        }
        scores = {rank: -0.1 for rank in range(16)}
        scores[2] = 1.5
        self.assertEqual(localize_scores(scores, thresholds)["reason"], "hysteresis_band")
        scores[2], scores[3] = 5.0, 3.0
        result = localize_scores(scores, thresholds)
        self.assertEqual(result["predicted"], "inconclusive")
        self.assertEqual(result["reason"], "top_runner_up_ratio_too_small")
        scores[3] = 1.0
        self.assertEqual(localize_scores(scores, thresholds)["predicted"], "rank:2")


if __name__ == "__main__":
    unittest.main()

