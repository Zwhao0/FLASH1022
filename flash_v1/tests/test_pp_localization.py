"""Localization invariants and optional replay of saved probe observations."""
import json
from pathlib import Path
import unittest

from flash_v1.analysis import localize


class LocalizationTests(unittest.TestCase):
    @staticmethod
    def rows(family='dense_DP', clique='a', count=100, values=((0, 10), (1, 190))):
        return [dict(tag=family, clique=clique, count=count, dtype=9,
                     rank=rank, duration_us=value) for rank, value in values]

    def test_legacy_and_explicit_shape_are_equivalent(self):
        rows = self.rows()
        legacy = [dict(row, tag=f"{row['tag']}:count{row['count']}:dtype{row['dtype']}")
                  for row in rows]
        for row in legacy:
            row.pop('count')
            row.pop('dtype')
        self.assertEqual(localize(rows), localize(legacy))

    def test_sizes_average_one_real_group_vote(self):
        high = self.rows()
        low = self.rows(count=200, values=((0, 75), (1, 125)))
        result = localize(high + low)
        self.assertAlmostEqual(result['candidates'][0]['score'], (.9 + .25) / 2)
        self.assertEqual(len(result['candidates'][0]['groups']), 1)
        self.assertEqual(result['candidates'][0]['tags'], ['dense_DP'])
        # Repeating the lower-score observation cannot add votes or change its weight.
        self.assertEqual(localize(high + low * 10), result)

    def test_stage1_compares_cliques_with_same_shape(self):
        rows = self.rows(clique='a', values=((0, 40), (1, 200)))
        rows += self.rows(clique='b', values=((2, 20), (3, 210)))
        result = localize(rows)
        self.assertIsNone(result['rank'])  # Maxima are too similar, despite rank separation.
        self.assertEqual(result['skipped'][0]['reason'], 'clique CV below threshold')
        self.assertTrue(result['skipped'][0]['group_screened'])

    def test_single_group_remains_eligible(self):
        result = localize(self.rows())
        self.assertEqual(result['rank'], 0)
        self.assertFalse(result['evidence'][0]['group_screened'])
        self.assertEqual(result['evidence'][0]['members'], [0, 1])

    def test_shape_must_cover_real_group_members(self):
        rows = [dict(row, group_members=[0, 1, 2, 3]) for row in self.rows()]
        result = localize(rows)
        self.assertIsNone(result['rank'])
        self.assertEqual(result['skipped'][0]['reason'], 'incomplete shape observations')

    def test_size_conflict_preserves_candidates_without_group_vote(self):
        rows = self.rows() + self.rows(count=200, values=((0, 190), (1, 10)))
        result = localize(rows)
        self.assertIsNone(result['rank'])
        self.assertEqual({c['rank'] for c in result['candidates']}, {0, 1})
        self.assertTrue(result['groups'][0]['conflict'])
        self.assertTrue(all(c['score'] == 0 for c in result['candidates']))

    def test_pipeline_conflicts_are_rank_permutation_invariant(self):
        for first, second, peer_a, peer_b in [(3, 7, 1, 5), (20, 2, 8, 11)]:
            rows = self.rows(clique='a', values=((first, 10), (peer_a, 190)))
            rows += self.rows(clique='b', count=200, values=((second, 20), (peer_b, 180)))
            topology = dict(groups=[dict(name='pipeline', members=[first, second]),
                                   dict(name='dense_world', label='dense_DP', members=[first, peer_a]),
                                   dict(name='dense_world', label='dense_DP', members=[second, peer_b])])
            result = localize(rows, topology=topology)
            self.assertEqual(result['status'], 'inconclusive')
            self.assertIsNone(result['rank'])
            self.assertEqual({c['rank'] for c in result['candidates']}, {first, second})
            self.assertIn('pipeline chain', result['reason'])
            fallback = localize([dict(r, pipeline_members=[first, second]) for r in rows])
            self.assertIsNone(fallback['rank'])

    def test_comparable_pipeline_screening_can_resolve(self):
        rows = self.rows(clique='a', values=((0, 10), (1, 400)))
        rows += self.rows(clique='b', values=((2, 80), (3, 100)))
        rows += self.rows(clique='b', count=200, values=((2, 20), (3, 180)))
        topology = dict(groups=[dict(name='pipeline', members=[0, 2]),
                               dict(name='dense_world', label='dense_DP', members=[0, 1]),
                               dict(name='dense_world', label='dense_DP', members=[2, 3])])
        result = localize(rows, topology=topology)
        self.assertEqual(result['rank'], 0)
        self.assertTrue(result['evidence'][0]['group_screened'])

    def test_saved_eight_rank_profile(self):
        folder = Path(__file__).resolve().parents[2] / 'results/stage3_pp_sustained'
        if not (folder / 'profile_1.json').exists():
            self.skipTest('saved probe profile unavailable')
        result = localize(json.loads((folder / 'profile_1.json').read_text()),
                          topology=json.loads((folder / 'topology.json').read_text()))
        self.assertEqual(result['status'], 'inconclusive')
        self.assertIsNone(result['rank'])
        self.assertEqual({c['rank'] for c in result['candidates']}, {3, 7})

    def test_saved_four_rank_size_conflict_does_not_promote_weak_vote(self):
        folder = Path(__file__).resolve().parents[2] / 'results/stage2_native_sustained'
        if not (folder / 'profile_1.json').exists():
            self.skipTest('saved probe profile unavailable')
        result = localize(json.loads((folder / 'profile_1.json').read_text()),
                          topology=json.loads((folder / 'topology.json').read_text()))
        self.assertEqual(result['status'], 'inconclusive')
        self.assertIsNone(result['rank'])
        self.assertEqual({c['rank'] for c in result['candidates']}, {2, 3})
        self.assertIn('conflicting sizes', result['reason'])

    def test_saved_four_rank_compute_profile(self):
        folder = Path(__file__).resolve().parents[2] / 'results/stage2_native_compute_both'
        if not (folder / 'profile_1.json').exists():
            self.skipTest('saved probe profile unavailable')
        result = localize(json.loads((folder / 'profile_1.json').read_text()),
                          topology=json.loads((folder / 'topology.json').read_text()))
        self.assertEqual(result['status'], 'identified')
        self.assertEqual(result['rank'], 3)


class DisabledReferenceTests(unittest.TestCase):
    def test_reference_mode_is_explicitly_disabled(self):
        with self.assertRaisesRegex(ValueError, 'temporarily disabled'):
            localize([], baseline_rows=[])


@unittest.skip('Healthy PROFILE differential localization disabled by user; retain experiments for reference')
class ReferenceLocalizationTests(unittest.TestCase):
    @staticmethod
    def samples(values=((0, 100), (1, 100)), count=100, dtype=9,
                clique='current', members=None, repeats=4):
        members = members or [r for r, _ in values]
        return [dict(tag='dense_DP', clique=clique, comm=f'{clique}-{rank}',
                     count=count, dtype=dtype, rank=rank, duration_us=value,
                     group_members=members, num_devices=len(members))
                for _ in range(repeats) for rank, value in values]

    def test_reference_matches_members_across_comm_and_clique_hashes(self):
        baseline = self.samples(clique='old-hash')
        current = self.samples(values=((0, 110), (1, 1010)), clique='new-hash')
        result = localize(current, baseline_rows=baseline)
        self.assertEqual(result['rank'], 0)
        self.assertEqual(result['evidence'][0]['delta_us'], {0: 10, 1: 910})
        self.assertEqual(result['evidence'][0]['score'], 900)

    def test_reference_never_crosses_count_or_dtype(self):
        baseline = self.samples(count=101) + self.samples(dtype=8)
        current = self.samples(values=((0, 100), (1, 1000)))
        result = localize(current, baseline_rows=baseline)
        self.assertIsNone(result['rank'])
        self.assertFalse(result['evidence'])

    def test_healthy_vs_healthy_and_noise_reject(self):
        baseline = self.samples()
        for current in [self.samples(), self.samples(values=((0, 101), (1, 99)))]:
            result = localize(current, baseline_rows=baseline)
            self.assertIsNone(result['rank'])
            self.assertFalse(result['evidence'])

    def test_reference_same_group_multiple_sizes_average_one_vote(self):
        baseline = self.samples() + self.samples(count=200)
        current = self.samples(values=((0, 100), (1, 1000)))
        current += self.samples(values=((0, 100), (1, 600)), count=200)
        result = localize(current, baseline_rows=baseline)
        self.assertEqual(result['rank'], 0)
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(result['candidates'][0]['score'], 700)
        self.assertEqual(len(result['candidates'][0]['groups']), 1)

    def test_reference_rank_permutation(self):
        baseline = self.samples(values=((20, 100), (2, 100)))
        current = self.samples(values=((20, 110), (2, 1010)))
        self.assertEqual(localize(current, baseline_rows=baseline)['rank'], 20)

    def test_uniform_added_wait_cancels_in_peer_difference(self):
        baseline = self.samples()
        current = self.samples(values=((0, 110), (1, 1010)))
        shifted = [dict(row, duration_us=row['duration_us'] + 5000) for row in current]
        first = localize(current, baseline_rows=baseline)
        second = localize(shifted, baseline_rows=baseline)
        self.assertEqual(second['rank'], first['rank'])
        self.assertEqual(second['candidates'][0]['score'], first['candidates'][0]['score'])

    def test_reference_incomplete_shape_does_not_crash_or_vote(self):
        baseline = self.samples()
        current = self.samples(values=((0, 100), (1, 1000)))
        # Another shape establishes rank 1 in this communicator, but shape 100 lacks it.
        current = [r for r in current if r['rank'] == 0]
        current += self.samples(values=((0, 100), (1, 100)), count=200)
        result = localize(current, baseline_rows=baseline)
        self.assertIsNone(result['rank'])
        self.assertFalse(result['evidence'])

    def test_reference_declared_members_cannot_be_inferred_from_partial_rows(self):
        members = [0, 1, 2, 3]
        baseline = self.samples(members=members)
        current = self.samples(values=((0, 100), (1, 1000)), members=members)
        result = localize(current, baseline_rows=baseline)
        self.assertIsNone(result['rank'])
        self.assertFalse(result['evidence'])

    def test_reference_incomplete_baseline_shape_does_not_crash_or_vote(self):
        baseline = [r for r in self.samples() if r['rank'] == 0]
        baseline += self.samples(count=200)
        current = self.samples(values=((0, 100), (1, 1000)))
        result = localize(current, baseline_rows=baseline)
        self.assertIsNone(result['rank'])
        self.assertFalse(result['evidence'])

    def test_actual_saved_healthy_self_has_no_added_wait(self):
        healthy_path = Path(__file__).resolve().parents[2] / 'results/stage3_pp_v2_healthy_profile/profile_1.json'
        if not healthy_path.exists():
            self.skipTest('fresh healthy probe profile unavailable')
        healthy = json.loads(healthy_path.read_text())
        result = localize(healthy, baseline_rows=healthy)
        self.assertIsNone(result['rank'])
        self.assertEqual(result['status'], 'inconclusive')
        self.assertFalse(result['evidence'])

    def test_saved_baseline_identifies_rank_in_old_and_independent_fault(self):
        root = Path(__file__).resolve().parents[2] / 'results'
        healthy = root / 'stage3_pp_v2_healthy_profile/profile_1.json'
        if not healthy.exists():
            self.skipTest('matched healthy profile unavailable')
        for name in ['stage3_pp_sustained', 'stage3_pp_v2_sustained']:
            with self.subTest(profile=name):
                folder = root / name
                if not (folder / 'profile_1.json').exists():
                    self.skipTest('saved fault profile unavailable')
                result = localize(json.loads((folder / 'profile_1.json').read_text()),
                                  baseline_rows=json.loads(healthy.read_text()),
                                  topology=json.loads((folder / 'topology.json').read_text()))
                self.assertEqual(result['rank'], 7)
                self.assertEqual(result['status'], 'identified')

    def test_reference_tied_minima_do_not_choose_by_rank_number(self):
        baseline = self.samples(values=((0, 100), (1, 100), (2, 100)))
        current = self.samples(values=((0, 100), (1, 100), (2, 1000)))
        result = localize(current, baseline_rows=baseline)
        self.assertIsNone(result['rank'])
        self.assertTrue(any(s['reason'] == 'reference-adjusted minimum tied' for s in result['skipped']))

    def test_reference_size_conflict_preserves_zero_score_candidates(self):
        baseline = self.samples() + self.samples(count=200)
        current = self.samples(values=((0, 100), (1, 1000)))
        current += self.samples(values=((0, 1000), (1, 100)), count=200)
        result = localize(current, baseline_rows=baseline)
        self.assertIsNone(result['rank'])
        self.assertEqual({c['rank'] for c in result['candidates']}, {0, 1})

    def test_reference_values_must_be_positive_and_finite(self):
        for invalid in [float('nan'), float('inf'), 0, -1]:
            for side in ['current', 'baseline']:
                current, baseline = self.samples(), self.samples()
                target = current if side == 'current' else baseline
                target[0]['duration_us'] = invalid
                with self.subTest(value=invalid, side=side):
                    with self.assertRaises(ValueError):
                        localize(current, baseline_rows=baseline)

    def test_reference_requires_repeated_samples(self):
        current = self.samples(values=((0, 100), (1, 1000)), repeats=1)
        result = localize(current, baseline_rows=self.samples())
        self.assertIsNone(result['rank'])
        self.assertFalse(result['evidence'])


if __name__ == '__main__':
    unittest.main()
