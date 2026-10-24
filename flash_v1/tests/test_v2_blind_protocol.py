"""Manual fixed fixtures only: tests never generate an actual random blind design."""
from pathlib import Path
import unittest
from unittest.mock import patch
from flash_v1.scripts import v2_blind_protocol as protocol


def fixture():
    gate=protocol.VerifiedFreeze(Path('/unused'),Path('/unused'),'a'*64,
        dict(model_file='candidate.json',whole_reference_sessions={layout:'existing_'+layout for layout in protocol.LAYOUTS}))
    layouts={}
    for side,layout in enumerate(protocol.LAYOUTS):
        cases=[dict(case_id=f'{layout}_{rank}',rank=rank,order=rank+1,
                    fault_mode=protocol.MODES[(rank+side)%5],target_degradation_pct=60.) for rank in range(16)]
        layouts[layout]=dict(cases=cases)
    design=dict(protocol='v2_blind32',freeze_manifest_sha256=gate.manifest_sha256,total_cases=32,seed=1,layouts=layouts)
    return gate,design


class ProtocolTests(unittest.TestCase):
    def test_no_random_sampling_before_verified_freeze(self):
        with patch.object(protocol.secrets,'randbits') as sample:
            with self.assertRaises(ValueError):protocol.build_design(None)
            sample.assert_not_called()
    def test_fixed_schedule_and_elapsed_only_strength(self):
        gate,design=fixture();layout='legacy';cases=design['layouts'][layout]['cases']
        doses={case['case_id']:100. for case in cases}
        with patch.object(protocol,'_gate'):
            plan=protocol.build_prevalidation(gate,design,layout,'manual_fixture',doses,1)
            self.assertEqual(plan['steps'],225)
            self.assertNotIn('profile_backend',plan)
            for flag in ('online','probe_enabled','profile_enabled','evidence_enabled'):self.assertIs(plan[flag],False)
            faults={step for interval in plan['intervals'] if interval['segment']=='fault' for step in range(interval['start'],interval['end']+1)}
            timing=[dict(iteration=step,wall_iteration_ms=150. if step in faults else 100.) for step in range(1,226)]
            measured=protocol.measure_prevalidation(plan,timing)
            self.assertTrue(measured['all_passed'])
            formal=protocol.build_formal(gate,design,layout,'manual_formal',measured,[measured])
            self.assertEqual(formal['steps'],1020)
            self.assertEqual(formal['profile_backend'],'semantic_only')
            self.assertIs(formal['probe_enabled'],False)
            self.assertIs(formal['evidence_enabled'],True)
            self.assertIs(formal['online'],True)
            self.assertIs(formal['profile_enabled'],False)
            self.assertEqual(sum(i['segment']=='fault' for i in formal['intervals']),16)
            self.assertEqual([i['rank'] for i in formal['intervals'] if i['segment']=='fault'],list(range(16)))
            with self.assertRaises(ValueError):protocol.measure_prevalidation(formal,timing)
    def test_dose_increase_only_capped_and_stops_after_three(self):
        row=dict(case_id='fixed',rank=3,fault_mode='whole_compute',target_degradation_pct=80.,dose=4.,measured_degradation_pct=10.,passed=False)
        measurement=dict(round_index=1,all_passed=False,by_case={'fixed':row})
        self.assertEqual(protocol.next_doses(measurement)['fixed'],8.)
        row['passed']=True;self.assertEqual(protocol.next_doses(measurement)['fixed'],4.)
        row['passed']=False;row['measured_degradation_pct']=0.
        with self.assertRaises(ValueError):protocol.next_doses(measurement)
        row['measured_degradation_pct']=10.;measurement['round_index']=3
        with self.assertRaises(ValueError):protocol.next_doses(measurement)
    def test_localizer_results_rejected_from_timing_input(self):
        with self.assertRaises(ValueError):protocol.measure_prevalidation(dict(online=False,probe_enabled=False,profile_enabled=False,evidence_enabled=False,measurement_only=True,role='strength_prevalidation'),[dict(iteration=1,wall_iteration_ms=1,predicted='rank:1')])

if __name__=='__main__':unittest.main()
