"""Replicated-activation TP for HF Qwen2MoE (transformers 4.53.2 SDPA).

Q/K/V and gate/up are column sharded; O/down are row sharded. Each TP
rank sees the same tokens. Embedding/head are vocabulary-row sharded; full
logits are gathered for the HF loss interface. Routers and norms remain replicated
and already receive complete gradients: do not sum those across TP.
No sequence parallelism, KV replication or cache support.
"""
import struct

import torch
import torch.distributed as dist
import torch.nn.functional as F


_collection_context = None


def set_collection_context(context):
    """Bind the two-uint64 collection mmap; None restores the ordinary TP path."""
    global _collection_context
    _collection_context = context


def _all_reduce_with_phase(value, group, phase):
    # Label only this actual TP collective, without extra autograd nodes, CUDA
    # events or synchronization. Preserve nested caller context on all exits.
    context = _collection_context
    if context is None or phase == 0:
        dist.all_reduce(value, group=group)
        return
    previous = struct.unpack_from('Q', context, 8)[0]
    struct.pack_into('Q', context, 8, phase)
    try:
        dist.all_reduce(value, group=group)
    finally:
        struct.pack_into('Q', context, 8, previous)


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, phase):
        ctx.group, ctx.phase = group, phase
        return value

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        _all_reduce_with_phase(gradient, ctx.group, ctx.phase)
        return gradient, None, None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, phase):
        value = value.contiguous().clone()
        _all_reduce_with_phase(value, group, phase)
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None


class ShardedLinear(torch.nn.Module):
    """Ordinary local GEMM with exactly one TP autograd boundary."""
    def __init__(self, original, group, dimension, phase_code=0):
        super().__init__()
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if original.weight.shape[dimension] % size:
            raise ValueError(f'Linear shape {tuple(original.weight.shape)} is not divisible by TP{size}')
        self.group, self.dimension = group, dimension
        self.forward_phase = phase_code
        self.backward_phase = phase_code + 1 if phase_code else 0
        self.in_features = original.in_features // (size if dimension == 1 else 1)
        self.out_features = original.out_features // (size if dimension == 0 else 1)
        self.weight = torch.nn.Parameter(original.weight.detach().chunk(size, dimension)[rank].clone())
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = dimension
        self.weight.partition_stride = 1
        if original.bias is None:
            self.register_parameter('bias', None)
        elif dimension == 0:
            self.bias = torch.nn.Parameter(original.bias.detach().chunk(size)[rank].clone())
            self.bias.tensor_model_parallel = True
            self.bias.partition_dim = 0
            self.bias.partition_stride = 1
        else:
            self.bias = torch.nn.Parameter(original.bias.detach().clone())

    def forward(self, value):
        if self.dimension == 0:
            return F.linear(_Copy.apply(value, self.group, self.backward_phase), self.weight, self.bias)
        value = _Reduce.apply(F.linear(value, self.weight), self.group, self.forward_phase)
        return value if self.bias is None else value + self.bias



class _GatherVocabulary(torch.autograd.Function):
    """Gather disjoint vocabulary columns; each rank evaluates the same loss.

    Backward takes only our column slice. Summing here would multiply the loss
    gradient by TP, since the complete loss is already replicated on every rank.
    """
    @staticmethod
    def forward(ctx, value, group):
        ctx.rank, ctx.size = dist.get_rank(group), dist.get_world_size(group)
        parts = [torch.empty_like(value) for _ in range(ctx.size)]
        dist.all_gather(parts, value.contiguous(), group=group)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.chunk(ctx.size, dim=-1)[ctx.rank].contiguous(), None


class VocabParallelEmbedding(torch.nn.Module):
    """Own a contiguous vocabulary interval; sum masked embedding outputs."""
    def __init__(self, original, group):
        super().__init__()
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if original.num_embeddings % size:
            raise ValueError('Vocabulary size must be divisible by TP')
        if original.max_norm is not None or original.scale_grad_by_freq or original.sparse:
            raise ValueError('Vocabulary TP requires ordinary dense embedding gradients')
        self.group = group
        self.num_embeddings = original.num_embeddings // size
        self.embedding_dim = original.embedding_dim
        self.vocab_start = rank * self.num_embeddings
        self.vocab_end = self.vocab_start + self.num_embeddings
        padding = original.padding_idx
        self.padding_idx = (padding - self.vocab_start if padding is not None and
                            self.vocab_start <= padding < self.vocab_end else None)
        self.weight = torch.nn.Parameter(original.weight.detach().chunk(size, 0)[rank].clone())
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 0
        self.weight.partition_stride = 1

    def forward(self, token_ids):
        outside = (token_ids < self.vocab_start) | (token_ids >= self.vocab_end)
        local_ids = (token_ids - self.vocab_start).masked_fill(outside, 0)
        local = F.embedding(local_ids, self.weight, padding_idx=self.padding_idx)
        local = local.masked_fill(outside.unsqueeze(-1), 0)
        return _Reduce.apply(local, self.group, 0)


class VocabParallelHead(ShardedLinear):
    """Column linear plus vocabulary gather, preserving HF full-logit output."""
    def __init__(self, original, group):
        super().__init__(original, group, dimension=0)

    def forward(self, hidden):
        return _GatherVocabulary.apply(super().forward(hidden), self.group)

def patch_tensor_parallel(model, group):
    """Patch after EP ownership pruning; preserves parameter names and HF forward.

    Returns a compact sharding manifest. A second application is rejected.
    Both query heads and KV heads must divide TP; no silent KV duplication.
    """
    size = dist.get_world_size(group)
    if size == 1:
        return dict(size=1, sharded_linears=0, replicated='embedding, head, routers, norms')
    if getattr(model, '_flash_tensor_parallel', False):
        raise ValueError('Tensor parallelism was already applied')
    cfg = getattr(model, 'hf_config', model.config)
    if getattr(cfg, '_attn_implementation', None) != 'sdpa':
        raise ValueError('Tensor parallel adapter requires HF SDPA')
    if cfg.num_attention_heads % size or cfg.num_key_value_heads % size:
        raise ValueError('Query and KV head counts must each be divisible by TP')
    if cfg.vocab_size % size:
        raise ValueError('Vocabulary size must be divisible by TP')
    decoder = model.decoder if hasattr(model, 'decoder') else model.model
    embedding = decoder.embed_tokens
    head = getattr(model, 'lm_head', None)
    tied = embedding is not None and head is not None and head.weight is embedding.weight
    if embedding is not None:
        decoder.embed_tokens = VocabParallelEmbedding(embedding, group)
    if head is not None:
        model.lm_head = VocabParallelHead(head, group)
        if tied:
            model.lm_head.weight = decoder.embed_tokens.weight
    count = 0
    for layer in decoder.layers:
        attention = layer.self_attn
        if attention.layer_idx is None or attention.layer_idx < 0:
            raise ValueError('Tensor parallel phase tags require original HF layer_idx')
        phase_base = (attention.layer_idx + 1) << 8
        if attention.__class__.__name__ != 'Qwen2MoeSdpaAttention':
            raise ValueError('Expected transformers 4.53.2 Qwen2MoeSdpaAttention')
        for name in ('q_proj', 'k_proj', 'v_proj'):
            setattr(attention, name, ShardedLinear(getattr(attention, name), group, 0, phase_base | 1))
            count += 1
        attention.o_proj = ShardedLinear(attention.o_proj, group, 1, phase_base | 1)
        count += 1
        attention.num_heads //= size
        attention.num_key_value_heads //= size
        attention.hidden_size //= size  # SDPA's final view uses the local Q width.
        # Q/KV ratio and head_dim stay unchanged.
        for module in list(layer.mlp.modules()):
            if module.__class__.__name__ != 'Qwen2MoeMLP':
                continue
            for name in ('gate_proj', 'up_proj'):
                setattr(module, name, ShardedLinear(getattr(module, name), group, 0, phase_base | 3))
                count += 1
            module.down_proj = ShardedLinear(module.down_proj, group, 1, phase_base | 3)
            module.intermediate_size //= size
            count += 1
    model._flash_tensor_parallel = True
    return dict(size=size, rank=dist.get_rank(group), sharded_linears=count,
                replicated='routers, norms', sequence_parallel=False,
                vocabulary_parallel=True, local_vocabulary_size=cfg.vocab_size // size,
                vocab_embedding=embedding is not None, vocab_head=head is not None)


def reference_shard(reference_tensor, parameter, group):
    """Select the full HF parameter/gradient slice owned by a local parameter."""
    if getattr(parameter, 'tensor_model_parallel', False):
        return reference_tensor.chunk(dist.get_world_size(group),
                                      parameter.partition_dim)[dist.get_rank(group)]
    return reference_tensor
