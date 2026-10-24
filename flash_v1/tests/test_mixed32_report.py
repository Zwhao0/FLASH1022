"""CPU fixtures for read-only mixed fault scoring."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from flash_v1.scripts.mixed32_report import ROOT, score_session, sha256


class MixedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.plan = dict(session_id='fixture', max_profiles=5, parallel_layout='legacy', intervals=[
            dict(id='healthy', start=1, end=15, target_ranks=[]),
            dict(id='fault', start=16, end=25, target_ranks=[3], delay_ms=10,
                 fault_mode='attention_bwd_compute', target_degradation_pct=10)])
        self.events = []
        self.training = [dict(iteration=i, time_ms=100 if i < 16 else 120,
                              wall_iteration_ms=100 if i < 16 else 120, profile_epoch=0, walltime=i)
                         for i in range(1, 26)]
        self.raw = []
        self.injections = [dict(iteration=i, rank=3, hook_calls=2, injected_cycles=0,
                                calibrated_injected_ms=20, engine='cuda_fp32_fma',
                                fault_mode='attention_bwd_compute') for i in range(16, 26)]

    def write(self, path, value, lines=False):
        path = self.path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))

    def score(self):
        self.write('plan.json', self.plan)
        self.write('host2/train.jsonl', self.training, True)
        self.write('host2/events.jsonl', self.events, True)
        for index, host in enumerate(('host2', 'host1')):
            self.write(f'{host}/node{index}_manifest.json', dict(status='completed',
                configuration={}, source_sha256={'fixture.py': 'abc'}, probe_sha256='p',
                snapshot_sha256='s', model_sha256={'model': 'm'}))
            self.write(f'{host}/drain_status.json', dict(status='complete', dropped=[]))
            (self.path / host / 'state.bin').write_bytes(b'\0' * 8)
            with gzip.open(self.path / host / 'raw_profile.jsonl.gz', 'wt') as stream:
                for row in self.raw if index == 0 else []:
                    stream.write(json.dumps(row) + '\n')
        self.write('host2/injection_rank3.jsonl', self.injections, True)
        return score_session(self.path, frozen_sources={'fixture.py': 'abc'},
                             frozen_model=sha256(ROOT / 'results/data16_models/wait_frozen.json'))

    def diagnose(self, predicted, epoch=1):
        self.events.extend([dict(event='detection', epoch=epoch, iteration=16 + epoch),
                            dict(event='localization', epoch=epoch, wall_time=24,
                                 result=dict(predicted=predicted))])

    def test_no_detection_and_real_compute_truth(self):
        report = self.score()
        case = report['fault_cases'][0]
        self.assertTrue(case['no_detection'])
        self.assertFalse(case['correct'])
        self.assertTrue(report['checks']['injection_truth'])
        self.assertEqual(report['injection_coverage'], 1)
        measured = case['measured_strength']['timing']['time_ms']['including_profile']
        self.assertEqual(measured['healthy']['iterations'], [13, 14, 15])
        self.assertAlmostEqual(measured['degradation_pct'], 20)

    def test_wrong_rejected_healthy_and_repeated_correct_are_all_kept(self):
        for epoch, prediction in enumerate(('rank:2', 'inconclusive', 'healthy', 'rank:3'), 1):
            self.diagnose(prediction, epoch)
        case = self.score()['fault_cases'][0]
        self.assertEqual(len(case['diagnoses']), 4)
        self.assertTrue(case['wrong_rank'] and case['rejected'] and case['judged_healthy'])
        self.assertFalse(case['correct'])
        self.assertEqual(case['detection_delay_iterations'], 1)

    def test_profile_truncation_and_excluding_profile(self):
        self.diagnose('rank:3')
        self.events.append(dict(event='profile_schedule', epoch=1, start=18, end=25))
        for row in self.training:
            if row['iteration'] >= 18:
                row['profile_epoch'] = 1
                row['time_ms'] = 200
        self.raw = [dict(iteration=i, rank=r, duration_us=1) for i in range(18, 25) for r in range(16)]
        report = self.score()
        self.assertFalse(report['profile_windows'][0]['complete_iteration_rank_coverage'])
        self.assertEqual(len(report['profile_windows'][0]['missing_timed_iteration_ranks']), 16)
        measured = report['fault_cases'][0]['measured_strength']['timing']['time_ms']['excluding_profile']
        self.assertEqual(measured['fault']['iterations'], [16, 17])
        self.assertAlmostEqual(measured['degradation_pct'], 20)

    def test_correct_localization_does_not_override_failed_integrity(self):
        self.diagnose('rank:3')
        report = self.score()
        self.assertTrue(report['fault_cases'][0]['correct'])
        self.assertFalse(report['integrity_passed'])
        self.assertFalse(report['checks']['detection_schedule_linkage'])

    def test_healthy_false_alarm_retained_and_corrupt_plan_does_not_crash(self):
        self.events.append(dict(event='detection', epoch=9, iteration=14))
        report = self.score()
        self.assertEqual(report['healthy_detection_count'], 1)
        self.write('plan.json', {'intervals': [None]})
        report = score_session(self.path)
        self.assertFalse(report['integrity_passed'])
        self.assertIn('evidence', report)
        self.assertIn('host2/train.jsonl', report['evidence'])

    def test_bad_engine_and_missing_injection_fail_integrity(self):
        self.injections.pop()
        self.injections[0]['engine'] = 'cuda_sleep'
        report = self.score()
        self.assertFalse(report['checks']['injection_exact_once_coverage'])
        self.assertFalse(report['checks']['injection_truth'])
        self.assertEqual(len(report['injection_violations']), 1)

    def test_missing_run_is_retained(self):
        report = score_session(self.path / 'missing')
        self.assertFalse(report['integrity_passed'])
        self.assertTrue(report['errors'])
        self.assertEqual(report['fault_denominator'], 0)


if __name__ == '__main__':
    unittest.main()
