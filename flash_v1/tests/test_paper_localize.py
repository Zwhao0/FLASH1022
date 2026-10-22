"""Synthetic baseline invariants; no GPU effectiveness claims."""
import json
from pathlib import Path
import tempfile
import unittest

from flash_v1.paper_localize import evaluate_dataset, localize


def rows(values, clique='a', tag='ppstage0_dp', count=100, kind='all_reduce'):
    members = [rank for rank, _ in values]
    return [dict(rank=rank, duration_us=value, clique=clique, tag=tag,
                 count=count, dtype=9, kind=kind, comm=str(rank),
                 group_members=members, group_name=tag, num_devices=len(members))
            for rank, value in values]


class PaperBaselineTests(unittest.TestCase):
    def test_maximum_clique_then_minimum_rank(self):
        data = rows([(0, 10), (1, 90)], 'low')
        data += rows([(2, 20), (3, 400)], 'high')
        result = localize(data)
        self.assertEqual(result['rank'], 2)
        self.assertEqual(result['evidence'][0]['clique'], 'high')
        self.assertTrue(result['evidence'][0]['group_screened'])

    def test_clique_cv_gate_precedes_rank_separation(self):
        data = rows([(0, 10), (1, 200)], 'a') + rows([(2, 20), (3, 210)], 'b')
        self.assertIsNone(localize(data)['rank'])
        self.assertEqual(localize(data)['skipped'][0]['reason'], 'clique CV below threshold')

    def test_stage_and_operation_comparability_do_not_pool(self):
        data = rows([(0, 10), (1, 100)], tag='ppstage0_dp')
        data += rows([(2, 95), (3, 100)], tag='ppstage1_dp')
        self.assertEqual(localize(data)['rank'], 0)
        data = rows([(0, 10), (1, 100)], kind='all_reduce')
        data += rows([(2, 95), (3, 100)], kind='reduce_scatter')
        self.assertEqual(localize(data)['rank'], 0)

    def test_multiple_shapes_contribute_one_maximum_without_global_veto(self):
        data = rows([(0, 10), (1, 190)])
        data += rows([(0, 125), (1, 75)], count=200)
        result = localize(data)
        self.assertEqual(result['rank'], 0)
        self.assertAlmostEqual(result['candidates'][0]['score'], .9)
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(localize(data + data)['candidates'], result['candidates'])

    def test_flat_or_tied_evidence_rejects_without_rank_number_bias(self):
        self.assertIsNone(localize(rows([(0, 100), (1, 100)]))['rank'])
        tied = rows([(0, 10), (1, 190)]) + rows([(0, 190), (1, 10)], count=200)
        self.assertIsNone(localize(tied)['rank'])
        self.assertIsNone(localize(rows([(0, 10), (1, 10), (2, 190)]))['rank'])

    def test_episode_scorer_has_fault_accuracy_rejection_and_healthy_fp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, truth, values in [('fault', 0, [(0, 10), (1, 190)]),
                                        ('healthy', None, [(0, 100), (1, 100)]),
                                        ('fp', None, [(0, 10), (1, 190)])]:
                folder = root / name
                folder.mkdir()
                profile = rows(values)
                topology = dict(world_size=2, groups=[dict(name='ppstage0_dp', label='ppstage0_dp',
                    localization=True, members=[0, 1], reporters=[0, 1])])
                for filename, value in [('manifest.json', dict(role='validation', session_id=name, fault_rank=truth)),
                                        ('profile.json', profile), ('topology.json', topology)]:
                    (folder / filename).write_text(json.dumps(value))
            report = evaluate_dataset(root)
            self.assertEqual(report['fault_localization_accuracy'], 1)
            self.assertEqual(report['inconclusive_runs'], 1)
            self.assertEqual(report['healthy_false_positives'], 1)
            self.assertEqual(report['healthy_runs'], 2)


if __name__ == '__main__':
    unittest.main()
