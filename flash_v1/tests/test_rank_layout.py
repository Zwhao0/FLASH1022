"""CPU-only checks against the actual vendor group construction and getters."""
import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch
from typing import Optional

from flash_v1.rank_layout import initialize_node_local, physical_rank


KWARGS = dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2,
              expert_model_parallel_size=2)
VENDOR = Path(__file__).resolve().parents[2] / 'Greyhound-master/Megatron-LM/megatron/core/parallel_state.py'


class FakeDist:
    group = types.SimpleNamespace(WORLD=object())

    def __init__(self, rank):
        self.rank = rank
        self.groups = []

    def is_initialized(self):
        return True

    is_available = is_initialized

    def get_world_size(self, group=None):
        return 16 if group is None else len(group)

    def get_rank(self, group=None):
        if group is None or group is self.group.WORLD:
            return self.rank
        return group.index(self.rank)

    def new_group(self, ranks=None, **kwargs):
        # Real new_group observes physical rank, even while the vendor sees logical rank.
        self.assert_physical_rank()
        members = tuple(sorted(range(16) if ranks is None else ranks))
        self.groups.append(members)
        return members if self.rank in members else None

    def assert_physical_rank(self):
        assert self.get_rank() == self.rank


def vendor_state(torch):
    tree = ast.parse(VENDOR.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    state = types.ModuleType('vendor_test_state')
    state.__dict__.update(torch=torch, Optional=Optional, os=os,
                          GlobalMemoryBuffer=lambda: object(),
                          log_model_parallel_to_redis=lambda *args: None)
    exec(compile(tree, str(VENDOR), 'exec'), state.__dict__)
    return state


class RankLayoutTests(unittest.TestCase):
    def test_mapping_is_permutation_and_inverse(self):
        self.assertEqual(sorted(map(physical_rank, range(16))), list(range(16)))
        for rank in range(16):
            self.assertEqual(physical_rank(physical_rank(rank)), rank)
        for rank in (-1, 16, 1.5):
            with self.assertRaises(ValueError):
                physical_rank(rank)

    @patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '8'})
    def test_all_vendor_groups_and_getters(self):
        calls = []
        for rank in range(16):
            dist = FakeDist(rank)
            torch = types.SimpleNamespace(distributed=dist)
            state = vendor_state(torch)
            initialize_node_local(state, torch, **KWARGS)
            self.assertIs(state.torch, torch)
            self.assertEqual(dist.get_rank(), rank)
            node, local = divmod(rank, 8)
            pp, et = divmod(local, 4)
            ep, tp = divmod(et, 2)
            expected_tp = [node * 8 + pp * 4 + ep * 2 + t for t in range(2)]
            expected_pp = [node * 8 + p * 4 + ep * 2 + tp for p in range(2)]
            expected_dp = [n * 8 + pp * 4 + e * 2 + tp for n in range(2) for e in range(2)]
            expected_expert_dp = [n * 8 + pp * 4 + ep * 2 + tp for n in range(2)]
            self.assertEqual(list(state.get_tensor_model_parallel_group()), expected_tp)
            self.assertEqual(list(state.get_pipeline_model_parallel_group()), expected_pp)
            self.assertEqual(list(state.get_data_parallel_group()), expected_dp)
            self.assertEqual(list(state.get_data_modulo_expert_parallel_group()), expected_expert_dp)
            self.assertEqual(state.get_tensor_model_parallel_rank(), tp)
            self.assertEqual(state.get_pipeline_model_parallel_rank(), pp)
            self.assertEqual(state.get_expert_model_parallel_rank(), ep)
            self.assertEqual(state.get_data_parallel_rank(), node * 2 + ep)
            self.assertEqual(state.get_data_modulo_expert_parallel_rank(), node)
            self.assertEqual(state.get_data_parallel_src_rank(), expected_dp[0])
            self.assertEqual(state.get_data_parallel_src_rank(True), expected_dp[0])
            self.assertEqual(state.get_tensor_model_parallel_src_rank(), expected_tp[0])
            self.assertEqual(state.get_pipeline_model_parallel_first_rank(), expected_pp[0])
            self.assertEqual(state.get_pipeline_model_parallel_last_rank(), expected_pp[-1])
            self.assertEqual(state.get_pipeline_model_parallel_next_rank(), expected_pp[1 - pp])
            self.assertEqual(state.get_pipeline_model_parallel_prev_rank(), expected_pp[1 - pp])
            self.assertEqual(state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS, expected_tp)
            self.assertEqual(state._DATA_PARALLEL_GLOBAL_RANKS, expected_dp)
            self.assertEqual(state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP, expected_dp)
            self.assertEqual(state._CONTEXT_PARALLEL_GLOBAL_RANKS, [rank])
            self.assertEqual(state._PIPELINE_GLOBAL_RANKS, expected_pp)
            self.assertEqual(state._EMBEDDING_GLOBAL_RANKS, expected_pp)
            self.assertEqual(state._POSITION_EMBEDDING_GLOBAL_RANKS, expected_pp[:1])
            self.assertTrue(state.is_rank_in_embedding_group())
            self.assertEqual(state.is_rank_in_position_embedding_group(), pp == 0)
            calls.append(dist.groups)
        self.assertTrue(all(groups == calls[0] for groups in calls))

    @patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '8'})
    def test_exception_restores_vendor_torch(self):
        torch = types.SimpleNamespace(distributed=FakeDist(4))
        state = types.SimpleNamespace(torch=torch)

        def fail(**kwargs):
            self.assertEqual(state.torch.distributed.get_rank(), 8)
            self.assertEqual(torch.distributed.get_rank(), 4)
            raise RuntimeError('injected failure')

        state.initialize_model_parallel = fail
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            initialize_node_local(state, torch, **KWARGS)
        self.assertIs(state.torch, torch)

    @patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '4'})
    def test_rejects_other_layout_before_initialization(self):
        torch = types.SimpleNamespace(distributed=FakeDist(0))
        state = vendor_state(torch)
        with self.assertRaises(ValueError):
            initialize_node_local(state, torch, **KWARGS)
        self.assertEqual(torch.distributed.groups, [])
        self.assertIs(state.torch, torch)


if __name__ == '__main__':
    unittest.main()
