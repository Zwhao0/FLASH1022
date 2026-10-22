"""Eight-GPU PP2/EP2/expert-DP2 equivalence check, with the native schedule.

Run from the repository root (not part of CPU unit tests):
  torchrun --standalone --nproc-per-node=8 -m flash_v1.tests.pipeline_verify --output /tmp/flash_pipeline_verify

Complete 24-layer tiny HF reference; tied embeddings, a final dense layer,
all-layer router auxiliary loss, and one then two microbatches are mandatory.
FP32 parameters/communication remove BF16 tied-gradient accumulation ambiguity.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_v1 import pipeline_adapter
from flash_v1.train import patch_moe


# FP32 isolates schedule/EP/scaling semantics from BF16 accumulation order.
# Dispatch GEMM shapes and sum-vs-concatenated-mean can still differ by FP32
# rounding. Relative-L2 also catches wrong scaling of small gradients.
TOLERANCES = dict(logits_atol=2e-6, logits_rtol=2e-5,
                  loss_atol=2e-6, loss_rtol=2e-6,
                  aux_atol=2e-6, aux_rtol=2e-6,
                  gradient_atol=2e-6, gradient_rtol=2e-4,
                  gradient_relative_l2=1e-3,
                  parameter_atol=2e-6, parameter_rtol=2e-5)


def reference_name(stage_name, pp_rank, layers_per_stage=12):
    match = re.match(r'decoder\.layers\.(\d+)\.(.*)', stage_name)
    if match:
        return f'model.layers.{int(match[1]) + pp_rank * layers_per_stage}.{match[2]}'
    if stage_name.startswith('decoder.'):
        return 'model.' + stage_name[len('decoder.'):]
    return stage_name


def errors(actual, expected):
    delta = actual.detach().float() - expected.detach().float()
    norm = expected.detach().float().norm().item()
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=delta.norm().item() / max(norm, 1e-12),
                reference_l2=norm)


def check_aux_formula(stage, router_logits, hf_aux):
    """Check packed all-layer auxiliary statistics against HF value and derivative.

    HF concatenates L*T router rows, then computes E*sum(mean(onehot,rows)
    * mean(softmax,rows)). Adding the two stages' integer counts and probability
    sums before dividing by L*T is the same formula, up to FP32 sum order.
    """
    routers = tuple(value for value in router_logits if value is not None)
    first_layers = 12  # First stage: 12 MoE; second stage: 11 MoE + dense.
    stats = stage._stats(routers[:first_layers], hf_aux.device)
    stats = stats + stage._stats(routers[first_layers:], hf_aux.device)
    count_end = stage.top_k * stage.num_experts
    counts = stats[:count_end].reshape(stage.top_k, stage.num_experts) / stats[-1]
    probabilities = stats[count_end:-1] / stats[-1]
    packed_aux = (counts * probabilities.unsqueeze(0)).sum() * stage.num_experts
    result = dict(value=errors(packed_aux, hf_aux), max_router_gradient_abs=0.0)
    torch.testing.assert_close(packed_aux, hf_aux,
                               atol=TOLERANCES['aux_atol'], rtol=TOLERANCES['aux_rtol'])
    expected_gradients = torch.autograd.grad(hf_aux, routers, retain_graph=True)
    packed_gradients = torch.autograd.grad(packed_aux, routers, retain_graph=True)
    for actual, expected in zip(packed_gradients, expected_gradients):
        result['max_router_gradient_abs'] = max(result['max_router_gradient_abs'],
                                                errors(actual, expected)['max_abs'])
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-5)
    return result


class AuditedSGD(torch.optim.SGD):
    """Observe the gradients actually consumed after the native finalizer."""
    def step(self, closure=None):
        self.consumed_gradients = {}
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    raise AssertionError('Stage-owned parameter has no finalized SGD gradient')
                self.consumed_gradients[id(parameter)] = parameter.grad.detach().clone()
        return super().step(closure)


def verify(output):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    rank = dist.get_rank()
    groups = pipeline_adapter.initialize_groups(output)
    dense = groups['dense']
    pp_rank = dist.get_rank(groups['pp'])
    if dist.get_world_size(dense) != 4 or dist.get_world_size(groups['ep']) != 2:
        raise AssertionError('Required layout is PP2 x EP2 x expert-DP2')
    if dist.get_world_size(groups['pp']) != 2 or dist.get_world_size(groups['expert_dp']) != 2:
        raise AssertionError('Required layout is PP2 x EP2 x expert-DP2')
    # A chain's dense rank identifies its data source independently of the
    # global-rank layout. Broadcast guarantees token identity along that chain.
    data_rank = dist.get_rank(dense)
    pp_members = dist.get_process_group_ranks(groups['pp'])
    config = Qwen2MoeConfig(
        vocab_size=64, hidden_size=16, intermediate_size=24,
        num_hidden_layers=24, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, moe_intermediate_size=8,
        shared_expert_intermediate_size=12, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True,
        decoder_sparse_step=1, mlp_only_layers=[23],
        tie_word_embeddings=True, router_aux_loss_coef=0.01,
        attention_dropout=0.0, use_cache=False,
    )
    config._attn_implementation = 'sdpa'
    torch.manual_seed(2026)
    reference = Qwen2MoeForCausalLM(config).cuda().float().train()
    if reference.lm_head.weight is not reference.model.embed_tokens.weight:
        raise AssertionError('Reference embedding and head must be tied')
    full_sharded = copy.deepcopy(reference)
    patched_layers = patch_moe(full_sharded, groups['ep'])
    if patched_layers != 23:
        raise AssertionError(f'Expected all 23 MoE layers plus the final dense layer, got {patched_layers}')
    stage = pipeline_adapter.StageModel(full_sharded, pp_rank, pp_size=2)
    if len(stage.decoder.layers) != 12:
        raise AssertionError('Both stages must retain all twelve assigned layers')
    if stage.config.bf16 or stage.config.params_dtype != torch.float32:
        raise AssertionError('This mechanism verification requires native FP32 parameters')
    if any(parameter.dtype != torch.float32 for parameter in stage.parameters()):
        raise AssertionError('Stage construction must preserve FP32 model weights')
    ddp = pipeline_adapter.wrap_stage(stage, groups, bucket_size=128)
    optimizer = AuditedSGD(stage.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    # Include the tied alias, while the reference optimizer visits it only once.
    expected_parameters = dict(reference.named_parameters(remove_duplicate=False))
    actual_parameters = dict(stage.named_parameters())
    names = {name: reference_name(name, pp_rank) for name in actual_parameters}
    if any(name not in expected_parameters for name in names.values()):
        raise AssertionError(f'Unmapped stage parameters: {names}')
    result = dict(rank=rank, pp_rank=pp_rank, data_rank=data_rank, status='running',
                  dtype='float32', tolerances=TOLERANCES, checks=[],
                  layers=24, moe_layers=23, tied_embeddings=True,
                  groups={key: dist.get_process_group_ranks(group) for key, group in groups.items()})
    path = output / f'pipeline_verify_rank{rank}.json'
    captured = []

    def observe_forward(module, inputs, outputs):
        if module.post_process:
            captured.append(tuple(value.detach().clone() for value in outputs))

    hook = stage.register_forward_hook(observe_forward)
    try:
        for step, microbatches in enumerate((1, 2), start=1):
            batch_list = []
            for microbatch in range(microbatches):
                generator = torch.Generator(device='cuda').manual_seed(
                    5000 + 100 * step + 10 * data_rank + microbatch)
                tokens = torch.randint(0, config.vocab_size, (1, 5),
                                       device='cuda', generator=generator)
                dist.broadcast(tokens, src=pp_members[0], group=groups['pp'])
                batch_list.append(dict(input_ids=tokens[:, :-1].contiguous(),
                                       next_token_labels=tokens[:, 1:].contiguous()))
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            aux_formula_checks = []
            for batch in batch_list:
                full = reference(input_ids=batch['input_ids'], output_router_logits=True,
                                 use_cache=False)
                if sum(logits is not None for logits in full.router_logits) != 23:
                    raise AssertionError('HF reference must include router outputs from all MoE layers')
                aux_formula_checks.append(check_aux_formula(stage, full.router_logits, full.aux_loss))
                ce = F.cross_entropy(full.logits.float().reshape(-1, config.vocab_size),
                                     batch['next_token_labels'].reshape(-1))
                loss = ce + config.router_aux_loss_coef * full.aux_loss
                expected_outputs.append((full.logits.detach().clone(),
                                         full.aux_loss.detach().clone(), loss.detach().clone()))
                (loss / microbatches).backward()
            # The unsharded reference computes each local batch once. Dense
            # DP4 averages every reference parameter, including all experts.
            # No PP reduction: each stage's full reference already has both
            # embedding/head paths and all layers; PP-chain batches coincide.
            for parameter in reference.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, group=dense)
                parameter.grad.div_(dist.get_world_size(dense))
            before = {name: parameter.detach().clone() for name, parameter in actual_parameters.items()}
            captured.clear()
            metrics = pipeline_adapter.run_step(ddp, batch_list, optimizer)
            check = dict(step=step, microbatches=microbatches,
                         outputs=[], gradients={}, parameters={}, metrics_count=len(metrics),
                         aux_formula_checks=aux_formula_checks,
                         max_update_abs=0.0)
            result['checks'].append(check)
            if stage.post_process:
                if len(captured) != microbatches or len(metrics) != microbatches:
                    raise AssertionError('Native schedule must run every requested microbatch')
                for index, ((logits, aux), (expected_logits, expected_aux, expected_loss)) in enumerate(
                        zip(captured, expected_outputs)):
                    actual_loss = F.cross_entropy(
                        logits.float().reshape(-1, config.vocab_size),
                        batch_list[index]['next_token_labels'].reshape(-1)) + config.router_aux_loss_coef * aux
                    record = dict(microbatch=index, logits=errors(logits, expected_logits),
                                  aux=errors(aux, expected_aux), loss=errors(actual_loss, expected_loss))
                    check['outputs'].append(record)
                    for kind, actual, expected in [('logits', logits, expected_logits),
                                                   ('aux', aux, expected_aux),
                                                   ('loss', actual_loss, expected_loss)]:
                        torch.testing.assert_close(actual.float(), expected.float(),
                                                   atol=TOLERANCES[f'{kind}_atol'],
                                                   rtol=TOLERANCES[f'{kind}_rtol'])
                    for metric_name, expected, kind in [('loss', expected_loss, 'loss'),
                                                        ('aux_loss', expected_aux, 'aux')]:
                        metric = metrics[index][metric_name]
                        check['outputs'][-1][f'schedule_{metric_name}'] = errors(metric, expected)
                        torch.testing.assert_close(metric.float(), expected.float(),
                                                   atol=TOLERANCES[f'{kind}_atol'],
                                                   rtol=TOLERANCES[f'{kind}_rtol'])
            # Gradients are snapshots from optimizer.step, after Megatron's
            # native finalizer has summed tied embedding/head gradients.
            for name, parameter in actual_parameters.items():
                expected = expected_parameters[names[name]]
                gradient = optimizer.consumed_gradients[id(parameter)]
                error = errors(gradient, expected.grad)
                check['gradients'][name] = dict(reference_name=names[name], **error)
                torch.testing.assert_close(gradient.float(), expected.grad.float(),
                                           atol=TOLERANCES['gradient_atol'],
                                           rtol=TOLERANCES['gradient_rtol'])
                if error['relative_l2'] > TOLERANCES['gradient_relative_l2']:
                    raise AssertionError(f'{name}: finalized gradient relative L2 error {error}')
            reference_optimizer.step()
            for name, parameter in actual_parameters.items():
                expected = expected_parameters[names[name]]
                check['parameters'][name] = errors(parameter, expected)
                check['max_update_abs'] = max(check['max_update_abs'],
                    (parameter.detach().float() - before[name].float()).abs().max().item())
                torch.testing.assert_close(parameter.float(), expected.float(),
                                           atol=TOLERANCES['parameter_atol'],
                                           rtol=TOLERANCES['parameter_rtol'])
            if check['max_update_abs'] == 0:
                raise AssertionError('The SGD verification must change stage-owned weights')
            path.write_text(json.dumps(result, indent=2))
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        hook.remove()
        path.write_text(json.dumps(result, indent=2))
    dist.barrier()
    if rank == 0:
        print('FP32 PP2 verification passed: full 24-layer HF outputs, aux, finalized gradients, '
              'tied embeddings, and SGD updates for one/two microbatches', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/flash_pipeline_verify'))
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    try:
        if dist.get_world_size() != 8:
            raise ValueError('This verification requires exactly eight GPU ranks')
        args.output.mkdir(parents=True, exist_ok=True)
        verify(args.output)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
