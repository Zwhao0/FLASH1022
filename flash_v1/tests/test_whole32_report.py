"""CPU-only whole32 scoring fixtures: no GPU imports or mutable inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from flash_v1.scripts.whole32_report import build_report, score_session


class WholeReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'formal'
        self.plan = dict(session_id='formal', parallel_layout='legacy', max_profiles=8,
            whole_compute_reference_session='pilot', intervals=[
                dict(id='healthy', start=1, end=15, target_ranks=[]),
                dict(id='fault', start=16, end=25, target_ranks=[3], fault_mode='whole_compute', slowdown_extra_ratio=2)])
        self.shapes = {phase: dict(op='aten.mm.default', phase=phase, median_ms=1., samples_ms=[1., 1., 1.], work_units=10)
                       for phase in ('fwd', 'bwd')}
        digest = hashlib.sha256(json.dumps(self.shapes, sort_keys=True).encode()).hexdigest()
        self.catalog = dict(catalog_sha256=digest, reference_catalog_sha256=digest, shapes=self.shapes,
                            quantum=dict(observed_ms=.05), fma_model=dict(units_per_ms=100., intercept_ms=0.))
        self.truth = [dict(iteration=i, rank=3, extra_ratio=2, compute_calls=2, healthy_estimated_ms=2.,
            requested_extra_ms=4., calibrated_injected_ms=4., injected_launches=2, pending_before_ms=0.,
            pending_extra_ms=0., pending_buckets={}, catalog_sha256=digest,
            operators=[dict(signature=p, op='aten.mm.default', phase=p, healthy_gpu_ms=1., calls=1,
                            requested_extra_ms=2., ratio=2, fallback=False) for p in ('fwd', 'bwd')],
            buckets={f'{p}:gemm': dict(requested_extra_ms=2., launch_rounds=200, injected_launches=1,
                                     calibrated_injected_ms=2.) for p in ('fwd', 'bwd')}, fallback_shapes=[])
            for i in range(16, 26)]
        self.events = []

    def write(self, path, data, lines=False):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r) + '\n' for r in data) if lines else json.dumps(data))

    def score(self):
        self.write('formal/plan.json', self.plan)
        self.write('pilot/plan.json', dict(parallel_layout='legacy'))
        self.write('formal/host2/events.jsonl', self.events, True)
        self.write('formal/host2/train.jsonl', [dict(iteration=i, time_ms=100 if i < 16 else 125,
            wall_iteration_ms=100 if i < 16 else 125, walltime=i, profile_epoch=0) for i in range(1, 26)], True)
        for rank in range(16):
            host = 'host2' if rank < 8 else 'host1'
            self.write(f'formal/{host}/whole_compute_calibration_rank{rank}.json', dict(self.catalog, rank=rank))
            self.write(f'pilot/{host}/whole_compute_calibration_rank{rank}.json', dict(self.catalog, rank=rank))
        for index, host in enumerate(('host2', 'host1')):
            self.write(f'formal/{host}/node{index}_manifest.json', dict(status='completed', plan=self.plan,
                online=True, probe_enabled=True, configuration={}, source_sha256={}, probe_sha256='p',
                model_sha256={}, snapshot_sha256='s', compute_workload_sha256='fma'))
            self.write(f'formal/{host}/drain_status.json', dict(status='complete', dropped=[], samples=0))
            (self.path / host / 'state.bin').write_bytes(b'\0' * 4)
            with gzip.open(self.path / host / 'raw_profile.jsonl.gz', 'wt'):
                pass
        self.write('formal/host2/whole_compute_truth_rank3.jsonl', self.truth, True)
        return score_session(self.path)

    def test_per_operator_truth_and_compact_output(self):
        report = self.score()
        self.assertTrue(report['checks']['injection_truth'])
        self.assertTrue(report['strict_measured_shape_coverage'])
        self.assertEqual(report['measured_shape_call_fraction'], 1)
        self.assertNotIn('operators', report['whole_compute_truth'][0])
        self.assertEqual(len(report['truth_files']), 1)
        self.assertTrue(report['fault_cases'][0]['strength_qualified_25pct'])

    def test_constant_total_cannot_hide_wrong_operator_ratio(self):
        self.truth[0]['operators'][0]['ratio'] = 3
        report = self.score()
        self.assertFalse(report['checks']['injection_truth'])
        self.assertIn('operator_timing_or_ratio', report['injection_violations'][0]['reasons'])

    def test_documented_fallback_is_valid_but_not_measured(self):
        row = self.truth[0]
        row['operators'][0].update(signature='new_shape', fallback=True)
        row['fallback_shapes'] = [dict(self.shapes['fwd'], signature='new_shape', calls=1,
            fallback='same_op_work_units_scaling', reference_signature='fwd', work_ratio=1.)]
        report = self.score()
        self.assertTrue(report['checks']['injection_truth'])
        self.assertFalse(report['strict_measured_shape_coverage'])
        self.assertLess(report['measured_shape_call_fraction'], 1)

    def test_unknown_shape_not_silently_skipped(self):
        self.truth[0]['operators'][0]['signature'] = 'unknown'
        report = self.score()
        self.assertFalse(report['checks']['injection_truth'])

    def test_detection_before_fault_does_not_count_correct(self):
        self.events = [dict(event='detection', epoch=1, iteration=14),
                       dict(event='localization', epoch=1, wall_time=24, result=dict(predicted='rank:3'))]
        report = self.score()
        self.assertEqual(report['healthy_detection_count'], 1)
        self.assertEqual(len(report['all_localizations']), 1)
        self.assertFalse(report['fault_cases'][0]['correct'])

    def test_repeated_disagreeing_diagnoses_retained(self):
        self.events = [dict(event='detection', epoch=1, iteration=16),
            dict(event='localization', epoch=1, wall_time=24, result=dict(predicted='rank:3')),
            dict(event='localization', epoch=1, wall_time=24, result=dict(predicted='rank:2'))]
        case = self.score()['fault_cases'][0]
        self.assertEqual(len(case['diagnoses']), 2)
        self.assertFalse(case['correct'])
        self.assertTrue(case['wrong_rank'])

    def test_fixed_denominator_and_read_only(self):
        self.score()
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = build_report([self.path, self.root / 'missing'], cleanup={})
        self.assertEqual(report['accuracy']['denominator'], 32)
        self.assertEqual(report['accuracy']['no_detection'], 32)
        self.assertEqual(len(report['cases']), 32)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse(report['integrity_passed'])

    def test_fma_totals_and_residual_cannot_hide_bad_launch_rounds(self):
        self.truth[0]['buckets']['fwd:gemm']['launch_rounds'] = 100
        report = self.score()
        self.assertFalse(report['checks']['injection_truth'])
        self.assertIn('bucket_fma_prediction', report['injection_violations'][0]['reasons'])

    def test_missing_truth_step_is_retained_as_missing(self):
        self.truth.pop()
        report = self.score()
        self.assertFalse(report['checks']['injection_exact_once_coverage'])
        self.assertEqual(report['injection_missing'], [(25, 3)])
        self.assertEqual(report['fault_denominator'], 1)

    def test_frozen_manifest_detects_artifact_tampering(self):
        self.score()
        self.write('frozen/formal_legacy.json', self.plan)
        digest = hashlib.sha256((self.root / 'frozen/formal_legacy.json').read_bytes()).hexdigest()
        self.write('frozen/freeze_manifest.json', dict(artifacts={'formal_legacy.json': digest}))
        self.write('frozen/formal_legacy.json', dict(self.plan, changed=True))
        report = build_report([self.path], cleanup={}, frozen_bundle=self.root / 'frozen')
        self.assertFalse(report['checks']['frozen_artifact:formal_legacy.json'])
        self.assertFalse(report['checks']['formal:frozen_formal_plan'])

    def test_duplicate_case_cannot_inflate_numerator(self):
        self.events = [dict(event='detection', epoch=1, iteration=16),
                       dict(event='localization', epoch=1, wall_time=24, result=dict(predicted='rank:3'))]
        self.score()
        report = build_report([self.path, self.path], cleanup={})
        self.assertEqual(report['accuracy']['correct'], 0)
        self.assertEqual(report['cases'][3]['evidence_case_count'], 2)


if __name__ == '__main__':
    unittest.main()
