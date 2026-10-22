"""Use GREYHOUND's unmodified Megatron groups, DDP buffers and backward hooks.

The HF checkpoint and verified token dispatcher remain the model implementation.
HF tensor sharding and the pipeline adapter use these groups and native schedules.
"""
import json
from pathlib import Path
import sys
import socket

import torch
import torch.distributed as dist


def core():
    source = Path(__file__).resolve().parents[1] / 'Greyhound-master/Megatron-LM'
    sys.path.insert(0, str(source))
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.transformer.transformer_config import TransformerConfig
    return parallel_state, DistributedDataParallel, TransformerConfig


def initialize_groups(output, expert_parallel_size=2, pipeline_parallel_size=1,
                      tensor_parallel_size=1):
    from flash_v1.topology import build_topology
    state, _, _ = core()
    # FLASH reports all actual group members below, synchronously. Replace only
    # the original fire-and-forget topology logger, not group construction.
    state.log_model_parallel_to_redis = lambda *args: None
    state.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size,
                                    expert_model_parallel_size=expert_parallel_size,
                                    pipeline_model_parallel_size=pipeline_parallel_size)
    dense = state.get_data_parallel_group()
    # GREYHOUND exposes only combined TP×EP. Dispatch must hold TP fixed.
    ep = None
    data_size = dist.get_world_size() // (tensor_parallel_size * pipeline_parallel_size)
    stage_size = data_size * tensor_parallel_size
    for stage in range(pipeline_parallel_size):
        for replica in range(data_size // expert_parallel_size):
            for tp_rank in range(tensor_parallel_size):
                members = [stage * stage_size +
                           (replica * expert_parallel_size + e) * tensor_parallel_size + tp_rank
                           for e in range(expert_parallel_size)]
                group = dist.new_group(members)
                if dist.get_rank() in members:
                    ep = group
    tp = state.get_tensor_model_parallel_group()
    pp_rank = state.get_pipeline_model_parallel_rank()
    expert_dp = state.get_data_modulo_expert_parallel_group()
    reports = []
    for name, label, group, stable in (
        ('dense_world', 'dense_DP', dense, True),
        ('expert_parallel', 'EP', ep, False),
        ('expert_dp', 'Expert_DP', expert_dp, True),
    ):
        reports.append(dict(name=name, label=f'{label}_PP{pp_rank}', pp_stage=pp_rank,
                            members=dist.get_process_group_ranks(group),
                            localization=stable))
    if tensor_parallel_size > 1:
        reports.append(dict(name='tensor_parallel', label=f'TP_PP{pp_rank}',
                            pp_stage=pp_rank, localization=True,
                            members=dist.get_process_group_ranks(tp)))
    if pipeline_parallel_size > 1:
        reports.append(dict(name='pipeline', label='PP', localization=False,
                            members=dist.get_process_group_ranks(state.get_pipeline_model_parallel_group())))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(rank=dist.get_rank(), host=socket.gethostname(),
        tp_rank=dist.get_rank(tp), pp_rank=pp_rank, ep_rank=dist.get_rank(ep),
        expert_dp_rank=dist.get_rank(expert_dp), data_rank=dist.get_rank(dense), groups=reports))
    if dist.get_rank() == 0:
        topology = build_topology(gathered, dist.get_world_size())
        topology['backend'] = 'GREYHOUND Megatron parallel_state'
        (output / 'topology.json').write_text(json.dumps(topology, indent=2))
    return ep, expert_dp, dense


def wrap(model, dense, expert_dp, bucket_size, config=None):
    _, ddp_class, config_class = core()
    for name, parameter in model.named_parameters():
        # Only routed experts are EP-sharded; shared experts and routers are DP.
        parameter.allreduce = '.experts.' not in name
    config = config or config_class(num_layers=model.config.num_hidden_layers,
                          hidden_size=model.config.hidden_size,
                          num_attention_heads=model.config.num_attention_heads,
                          tensor_model_parallel_size=core()[0].get_tensor_model_parallel_world_size(),
                          bf16=True)
    return ddp_class(config, model, dense,
                     accumulate_allreduce_grads_in_fp32=False,
                     overlap_grad_reduce=True, use_distributed_optimizer=False,
                     expert_data_parallel_group=expert_dp, bucket_size=bucket_size)


def finish_gradients(ddp):
    ddp.finish_grad_sync()
    # Megatron's hooks clear .grad. Plain SGD must use the reduced main_grad.
    for parameter in ddp.module.parameters():
        parameter.grad = parameter.main_grad


def describe(ddp):
    return [dict(label=label, buckets=len(buffer.buckets),
                 elements=buffer.numel,
                 group=dist.get_process_group_ranks(buffer.data_parallel_group),
                 scaling=buffer.gradient_scaling_factor)
            for label, buffers in [('dense_DP', ddp.buffers),
                                   ('Expert_DP', ddp.expert_parallel_buffers)]
            for buffer in buffers]
