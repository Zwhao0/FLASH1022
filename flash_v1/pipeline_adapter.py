"""Experimental HF Qwen MoE PP2 with unmodified Megatron 1F1B and DDP.

Fixed length, unpadded, no-cache training only. Initialize Megatron PP2/EP2,
build StageModel(full_hf_model, pp_rank), then wrap with its native .config. Both stages must
receive identical batches within each PP chain. GPU equivalence is not yet
verified; FP32 sufficient statistics can differ from HF reduction rounding.
"""
import functools
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from flash_v1.megatron_adapter import core


def initialize_groups(output, tensor_parallel_size=1):
    from flash_v1.megatron_adapter import initialize_groups as initialize
    state, _, _ = core()
    if torch.distributed.get_world_size() != 8 * tensor_parallel_size:
        raise ValueError('TP x PP2 x EP2 x ExpertDP2 requires 8*TP ranks')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ep, expert_dp, dense = initialize(output, expert_parallel_size=2,
                                      pipeline_parallel_size=2, tensor_parallel_size=tensor_parallel_size)
    return dict(ep=ep, expert_dp=expert_dp, dense=dense,
                pp=state.get_pipeline_model_parallel_group(),
                tp=state.get_tensor_model_parallel_group())


class StageModel(torch.nn.Module):
    """Consumes the full model; do not reuse it after constructing this stage."""
    def __init__(self, full_hf_model, pp_rank, pp_size=2):
        super().__init__()
        _, _, config_class = core()
        from megatron.core.enums import ModelType
        cfg = full_hf_model.config
        if pp_size != 2 or pp_rank not in (0, 1) or cfg.num_hidden_layers % 2:
            raise ValueError('This adapter requires complete even-layer models and PP2')
        if getattr(cfg, '_attn_implementation', None) != 'sdpa':
            raise ValueError('Use the previously verified HF SDPA model')
        self.hf_config = cfg
        self.parameter_dtype = next(full_hf_model.parameters()).dtype
        if self.parameter_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('Only FP32 verification and BF16 training are supported')
        self.pre_process = pp_rank == 0
        self.post_process = pp_rank == 1
        self.model_type = ModelType.encoder_or_decoder
        self.share_embeddings_and_output_weights = bool(cfg.tie_word_embeddings)
        self.config = config_class(
            num_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_moe_experts=cfg.num_experts,
            pipeline_model_parallel_size=2, expert_model_parallel_size=1,
            tensor_model_parallel_size=core()[0].get_tensor_model_parallel_world_size(),
            bf16=self.parameter_dtype == torch.bfloat16,
            params_dtype=self.parameter_dtype, pipeline_dtype=torch.float32,
            overlap_p2p_comm=False, batch_p2p_comm=True,
            deallocate_pipeline_outputs=False)
        # The native Megatron MoE dispatcher requires SP for TP+EP. FLASH uses
        # HF replicated activations with independent fixed-TP EP dispatch instead;
        # its TP autograd boundaries sum partial input gradients themselves. Build
        # with EP1 to bypass only that dispatcher-specific guard, then restore the
        # actual group size. Keep SP false: enabling it would shrink PP payloads
        # and wrongly sum gradients of our replicated norms.
        actual_ep_size = core()[0].get_expert_model_parallel_world_size()
        if cfg.num_experts is None or cfg.num_experts <= 0 or cfg.num_experts % actual_ep_size:
            raise ValueError('HF expert count must be positive and divisible by actual EP')
        self.config.expert_model_parallel_size = actual_ep_size
        self.config.flash_custom_moe_dispatcher = True
        self.decoder = full_hf_model.model
        local_layers = cfg.num_hidden_layers // 2
        start = pp_rank * local_layers
        self.layer_start = start
        self.decoder.layers = torch.nn.ModuleList(list(self.decoder.layers)[start:start + local_layers])
        self.lm_head = full_hf_model.lm_head if self.post_process else None
        if self.pre_process:
            self.decoder.norm = torch.nn.Identity()
        else:
            self.decoder.embed_tokens = None
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.stats_size = (self.top_k + 1) * self.num_experts + 1
        self.input_tensor = None

    def set_input_tensor(self, tensors):
        self.input_tensor = tensors[0] if isinstance(tensors, list) else tensors

    def shared_embedding_or_output_weight(self):
        return (self.decoder.embed_tokens.weight if self.pre_process
                else self.lm_head.weight)

    def extra_positions(self, batch_size):
        return math.ceil(self.stats_size / (batch_size * self.hf_config.hidden_size))

    def _stats(self, router_logits, device):
        # Same sufficient statistics as transformers 4.53's full-layer formula.
        counts = torch.zeros(self.top_k, self.num_experts, device=device)
        probability_sum = torch.zeros(self.num_experts, device=device)
        total = 0
        for logits in router_logits:
            probabilities = F.softmax(logits, dim=-1)
            selected = torch.topk(probabilities, self.top_k, dim=-1).indices
            counts = counts + F.one_hot(selected, self.num_experts).float().sum(0)
            probability_sum = probability_sum + probabilities.float().sum(0)
            total += logits.shape[0]
        return torch.cat((counts.flatten(), probability_sum,
                          torch.tensor([total], dtype=torch.float32, device=device)))

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        if self.pre_process:
            hidden = self.decoder.embed_tokens(input_ids)
            previous_stats = None
        else:
            if self.input_tensor is None:
                raise RuntimeError('Megatron must supply the previous stage tensor')
            expected = sequence_length + self.extra_positions(batch_size)
            if self.input_tensor.shape != (expected, batch_size, self.hf_config.hidden_size):
                raise ValueError('Pipeline shape/batch mismatch')
            hidden = self.input_tensor[:sequence_length].transpose(0, 1).to(self.parameter_dtype)
            previous_stats = self.input_tensor[sequence_length:].reshape(-1)[:self.stats_size]
        positions = torch.arange(sequence_length, device=hidden.device)
        position_ids = positions.unsqueeze(0)
        mask = self.decoder._update_causal_mask(None, hidden, positions, None, False)
        position_embeddings = self.decoder.rotary_emb(hidden, position_ids)
        routers = []
        for layer in self.decoder.layers:
            outputs = layer(hidden, attention_mask=mask, position_ids=position_ids,
                            past_key_value=None, output_attentions=False,
                            output_router_logits=True, use_cache=False,
                            cache_position=positions, position_embeddings=position_embeddings)
            hidden = outputs[0]
            if outputs[-1] is not None:
                routers.append(outputs[-1])
        stats = self._stats(routers, hidden.device)
        if self.pre_process:
            capacity = self.extra_positions(batch_size) * batch_size * hidden.shape[-1]
            packed = F.pad(stats, (0, capacity - self.stats_size)).reshape(
                -1, batch_size, hidden.shape[-1])
            return torch.cat((hidden.transpose(0, 1).float(), packed), dim=0)
        stats = stats + previous_stats
        count_end = self.top_k * self.num_experts
        denominator = stats[-1]
        counts = stats[:count_end].reshape(self.top_k, self.num_experts) / denominator
        # HF softmax and mean preserve the router logits/model dtype.
        probabilities = (stats[count_end:-1] / denominator).to(hidden.dtype)
        aux_loss = (counts * probabilities.unsqueeze(0)).sum() * self.num_experts
        logits = self.lm_head(self.decoder.norm(hidden))
        return logits, aux_loss


def _loss(logits, *, batch, aux, coefficient):
    if 'next_token_labels' in batch:
        labels = batch['next_token_labels']
    else:
        labels = batch.get('labels', batch['input_ids'])[:, 1:]
        logits = logits[:, :-1]
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
    loss = ce + coefficient * aux
    return loss, {'loss': loss.detach(), 'cross_entropy': ce.detach(), 'aux_loss': aux.detach()}


def forward_step(data_iterator, model):
    """Megatron schedule callback; the input batch is local to each PP stage.

    input_ids are unpadded fixed-shape tokens. next_token_labels uses FLASH's
    external shift; labels (or omitted labels) uses HF's internal causal shift.
    All stages need the same batch shape and PP-chain token/label identity.
    """
    stage = model
    while not isinstance(stage, StageModel):
        stage = stage.module
    batch = next(data_iterator)
    if 'attention_mask' in batch:
        raise ValueError('Padding/masked batches are not supported')
    result = model(batch['input_ids'])
    if stage.post_process:
        logits, aux = result
        return logits, functools.partial(_loss, batch=batch, aux=aux,
                                          coefficient=stage.hf_config.router_aux_loss_coef)
    return result, None


StageWrapper = StageModel


def wrap_stage(stage, groups, bucket_size=2_000_000):
    from flash_v1.megatron_adapter import wrap
    from megatron.core.distributed import finalize_model_grads
    ddp = wrap(stage, groups['dense'], groups['expert_dp'], bucket_size,
               config=stage.config)
    stage.config.no_sync_func = ddp.no_sync
    stage.config.finalize_model_grads_func = finalize_model_grads
    return ddp


def run_step(ddp, batches, optimizer):
    """One native 1F1B/SGD step; returns last-stage detached microbatch metrics."""
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    if not batches:
        raise ValueError('At least one microbatch is required')
    stage = ddp.module
    shape = batches[0]['input_ids'].shape
    for batch in batches:
        if 'attention_mask' in batch or batch['input_ids'].shape != shape:
            raise ValueError('Only fixed-shape unpadded batches are supported')
    optimizer.zero_grad(set_to_none=True)
    ddp.zero_grad_buffer()
    metrics = get_forward_backward_func()(
        forward_step_func=forward_step, data_iterator=iter(batches), model=ddp,
        num_microbatches=len(batches),
        seq_length=shape[1] + stage.extra_positions(shape[0]), micro_batch_size=shape[0])
    # Native finalizer already waited DP buckets and summed tied PP gradients.
    for parameter in stage.parameters():
        parameter.grad = parameter.main_grad
    optimizer.step()
    return metrics
