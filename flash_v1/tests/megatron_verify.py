"""Four-GPU native Megatron overlap check; no checkpoint or Redis required.

Run from the repository root:
  torchrun --standalone --nproc-per-node=4 -m flash_v1.tests.megatron_verify --output /tmp/flash_megatron_verify
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from flash_v1 import megatron_adapter
from flash_v1.train import patch_block


class BlockModel(torch.nn.Module):
    def __init__(self, config, block):
        super().__init__()
        self.config = config
        self.block = block

    def forward(self, x):
        return self.block(x)


def verify(output):
    from transformers import Qwen2MoeConfig
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

    rank = dist.get_rank()
    ep, expert_dp, dense = megatron_adapter.initialize_groups(output)
    config = Qwen2MoeConfig(
        hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=12,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
    )
    torch.manual_seed(2026)
    reference = BlockModel(config, Qwen2MoeSparseMoeBlock(config)).cuda().float().train()
    sharded = copy.deepcopy(reference)
    patch_block(sharded.block, ep)
    # Small buckets force several real asynchronous reductions per backward.
    ddp = megatron_adapter.wrap(sharded, dense, expert_dp, bucket_size=128)
    actual_parameters = dict(sharded.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    optimizer = torch.optim.SGD(sharded.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    topology = megatron_adapter.describe(ddp)
    records = []
    for step, tokens in enumerate((1, 7), start=1):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        ddp.zero_grad_buffer()
        # Identical single-token input across EP sources guarantees at least
        # two globally unused experts; the second step uses distinct rank data.
        torch.manual_seed(1000 + tokens + (rank if tokens > 1 else 0))
        x = torch.randn(1, tokens, 16, device="cuda", requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        actual, actual_router = ddp(x)
        expected, expected_router = reference(x_ref)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual_router, expected_router, atol=2e-6, rtol=2e-5)
        (actual.square().mean() + actual_router.square().mean() * 0.01).backward()
        (expected.square().mean() + expected_router.square().mean() * 0.01).backward()
        torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-6, rtol=2e-5)
        buckets = [bucket for buffer in ddp.buffers + ddp.expert_parallel_buffers
                   for bucket in buffer.buckets]
        assert buckets and all(bucket.communication_issued for bucket in buckets), (
            "Every bucket must launch from backward readiness hooks before finish_gradients"
        )
        assert all(bucket.communication_handle is not None for bucket in buckets)
        megatron_adapter.finish_gradients(ddp)
        maximum_gradient_error = 0.0
        unused_experts = set()
        for name, parameter in reference_parameters.items():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, group=dense)
            parameter.grad.div_(dist.get_world_size(dense))
            if '.experts.' in name and parameter.grad.count_nonzero().item() == 0:
                unused_experts.add(int(name.split('.experts.')[1].split('.')[0]))
            if name in actual_parameters:
                main_grad = actual_parameters[name].main_grad
                torch.testing.assert_close(main_grad, parameter.grad, atol=2e-6, rtol=2e-5)
                maximum_gradient_error = max(
                    maximum_gradient_error,
                    (main_grad - parameter.grad).abs().max().item(),
                )
        if tokens == 1:
            assert len(unused_experts) >= 2, "Single-token case must exercise unused experts"
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in actual_parameters.items():
            torch.testing.assert_close(parameter, reference_parameters[name], atol=2e-6, rtol=2e-5)
        records.append(dict(step=step, tokens=tokens, buckets_issued=len(buckets),
                            unused_experts=sorted(unused_experts),
                            max_gradient_error=maximum_gradient_error))
    torch.cuda.synchronize()
    result = dict(rank=rank, status="passed", checks=records, buffers=topology)
    (output / f"verify_rank{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()
    if rank == 0:
        print("Megatron overlap verified: HF outputs/input gradients, world-mean owned gradients, two SGD updates, unused experts, all buckets issued", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/flash_megatron_verify'))
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    # Dispatch changes GEMM batch shapes. Compare full FP32, not shape-dependent
    # TF32 rounding, when validating the autograd/gradient scaling semantics.
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl')
    try:
        if dist.get_world_size() != 4:
            raise ValueError('This verification requires exactly four GPU ranks')
        args.output.mkdir(parents=True, exist_ok=True)
        verify(args.output)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
