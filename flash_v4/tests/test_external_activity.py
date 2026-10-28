import unittest

from flash_v4.external_activity import vllm_activity


class ExternalActivityTests(unittest.TestCase):
    def test_apps_resolves_truncated_pmon_name_and_ignores_own_mps(self):
        apps = dict(returncode=0, stdout='GPU-x, 123, VLLM::Worker_PP0_TP0_EP0, 64000\n')
        pmon = dict(returncode=0, stdout='# gpu pid type sm mem command\n0 123 C 25 0 worker\n0 456 M 99 0 nvidia-cuda-mps\n')
        result = vllm_activity(apps, pmon)
        self.assertTrue(result['activity_observed'])
        self.assertEqual(len(result['active_worker_rows']), 1)
        self.assertEqual(result['active_worker_rows'][0]['pid'], 123)

    def test_idle_and_missing_metrics_do_not_prove_absence(self):
        for stdout in ['0 123 C 0 0 VLLM::Worker\n', '0 123 C - - VLLM::Worker\n', '0 - - - - -\n']:
            result = vllm_activity({}, dict(returncode=0, stdout=stdout))
            self.assertFalse(result['activity_observed'])
            self.assertIn('not proof', result['interpretation'])

    def test_failed_pmon_is_unavailable(self):
        result = vllm_activity({}, dict(returncode=1, stdout='0 123 C 90 0 VLLM::Worker'))
        self.assertFalse(result['pmon_available'])
        self.assertEqual(result['measured_worker_rows'], 0)


if __name__ == '__main__':
    unittest.main()
