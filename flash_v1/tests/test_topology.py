import unittest
from pathlib import Path
import tempfile

from flash_v1.topology import build_topology, label_cliques, load_topology


def topology(*groups, world_size=8):
    reports = [dict(rank=r, groups=[g for g in groups if r in g['members']])
               for r in range(world_size)]
    return build_topology(reports, world_size)


def group(name, members, label='Expert_DP', localization=True):
    return dict(name=name, members=members, label=label, localization=localization)


def rows(clique, members, n=None):
    return [dict(clique=clique, rank=r, num_devices=n or len(members),
                 count=1024, dtype=7, duration_us=100, comm=str(r), group_id=1)
            for r in members]


class TopologyTests(unittest.TestCase):
    def test_actual_members_and_equal_size_different_semantics(self):
        t = topology(group('expert_dp', [1, 5]),
                     group('expert_parallel', [1, 3], None, False))
        accepted, diagnostics = label_cliques(rows('dp', [1, 5]) + rows('ep', [1, 3]), t)
        self.assertEqual({r['rank'] for r in accepted}, {1, 5})
        self.assertEqual({r['tag'] for r in accepted}, {'Expert_DP'})
        self.assertTrue(all(r['count'] == 1024 and r['dtype'] == 7 for r in accepted))
        self.assertTrue(all(r['group_members'] == [1, 5] for r in accepted))
        self.assertEqual(diagnostics[0]['reason'], 'group excluded from localization')

    def test_missing_rank_and_inconsistent_size_are_excluded(self):
        t = topology(group('expert_dp', [1, 5]))
        accepted, diagnostics = label_cliques(rows('partial', [1], 2), t)
        self.assertEqual(accepted, [])
        self.assertEqual(diagnostics[0]['observed_members'], [1])
        self.assertEqual(diagnostics[0]['reason'], 'incomplete or inconsistent clique')
        observations = rows('bad_size', [1, 5])
        observations[1]['num_devices'] = 3
        self.assertEqual(label_cliques(observations, t)[0], [])

    def test_ambiguous_and_incomplete_topology_are_excluded(self):
        t = topology(group('expert_dp', [1, 5]), group('tensor', [1, 5], None, False))
        accepted, diagnostics = label_cliques(rows('alias', [1, 5]), t)
        self.assertEqual(accepted, [])
        self.assertEqual(diagnostics[0]['reason'], 'ambiguous topology match')
        t = topology(group('expert_dp', [1, 5]))
        t['groups'][0]['reporters'] = [1]
        accepted, diagnostics = label_cliques(rows('dp', [1, 5]), t)
        self.assertEqual(accepted, [])
        self.assertEqual(diagnostics[0]['reason'], 'invalid or incomplete topology group')

    def test_rank_report_semantic_disagreement_is_not_silently_merged(self):
        reports = [dict(rank=1, groups=[group('dp', [1, 5])]),
                   dict(rank=5, groups=[group('dp', [1, 5], None, False)])]
        t = build_topology(reports, 8)
        self.assertEqual(len(t['groups']), 2)
        self.assertEqual(label_cliques(rows('conflict', [1, 5]), t)[0], [])

    def test_missing_or_legacy_topology_does_not_infer_from_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'topology.json'
            self.assertIsNone(load_topology(path)[0])
            path.write_text('{"world_size":4,"dense_world":[0,1,2,3]}')
            self.assertIsNone(load_topology(path)[0])
        self.assertEqual(label_cliques(rows('four', [0, 1, 2, 3]), None)[0], [])

    def test_phase_one_tags_remain_compatible(self):
        t = topology(group('dense_world', [0, 1, 2, 3], 'dense_DP'),
                     group('expert_dp', [0, 2]), group('expert_dp', [1, 3]), world_size=4)
        accepted, diagnostics = label_cliques(rows('dense', [0, 1, 2, 3]) + rows('expert', [1, 3]), t)
        self.assertEqual(diagnostics, [])
        self.assertEqual({r['tag'] for r in accepted},
                         {'dense_DP', 'Expert_DP'})


if __name__ == '__main__':
    unittest.main()
