"""Full pretrained Qwen2.5-0.5B-MoE training: torchrun --nproc-per-node=4 train.py.

Ranks [0,1] and [2,3] dispatch tokens (EP=2); [0,2] and [1,3]
replicate the corresponding experts (expert DP=2). No layers are removed.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import socket
import time
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_v1.topology import build_topology

REVISION = "4b89028ef7376d0e92acbbc4a7f560c2374aeec0"
MODEL_ID = "bayliner1980/Qwen2.5-0.5B-MoE-A0.3B"


class VariableAllToAll(torch.autograd.Function):
    """A variable token exchange whose adjoint is the reverse exchange."""
    @staticmethod
    def forward(ctx, x, send, receive, group):
        ctx.send, ctx.receive, ctx.group = send, receive, group
        out = x.new_empty((sum(receive), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), receive, send, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((sum(ctx.send), *grad.shape[1:]))
        dist.all_to_all_single(out, grad.contiguous(), ctx.send, ctx.receive,
                               group=ctx.group)
        return out, None, None, None


def ep_forward(self, x):
    shape = x.shape
    flat_x = x.reshape(-1, shape[-1])
    router_logits = self.gate(flat_x)
    scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    weights, indices = torch.topk(scores, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights.to(x.dtype)
    flat_indices = indices.reshape(-1)
    # Contiguous expert ownership; stable sorting keeps the return permutation.
    owners = flat_indices // self._experts_per_rank
    order = torch.argsort(owners, stable=True)
    send_count = torch.bincount(owners, minlength=self._ep_size)
    receive_count = torch.empty_like(send_count)
    dist.all_to_all_single(receive_count, send_count, group=self._ep_group)
    send, receive = send_count.tolist(), receive_count.tolist()
    dispatched = flat_x[order // indices.shape[-1]]
    received = VariableAllToAll.apply(dispatched, send, receive, self._ep_group)
    received_ids = flat_indices.new_empty(sum(receive))
    dist.all_to_all_single(received_ids, flat_indices[order].contiguous(),
                           receive, send, group=self._ep_group)
    # Keep every owned expert and both exchanges in every rank's autograd graph,
    # including when an expert (or this rank) receives zero tokens.
    expert_output = received * 0
    first = self._ep_index * self._experts_per_rank
    for expert_id in range(first, first + self._experts_per_rank):
        positions = (received_ids == expert_id).nonzero(as_tuple=True)[0]
        values = self.experts[expert_id](received[positions])
        expert_output = expert_output.index_add(0, positions, values)
    returned = VariableAllToAll.apply(expert_output, receive, send, self._ep_group)
    restored = returned[torch.argsort(order)]
    result = (restored.reshape(*weights.shape, shape[-1]) *
              weights.unsqueeze(-1)).sum(1).reshape(shape)
    shared = self.shared_expert(flat_x) * torch.sigmoid(self.shared_expert_gate(flat_x))
    return result + shared.reshape(shape), router_logits


def create_topology(output):
    if dist.get_world_size() != 4:
        raise ValueError("Run exactly four ranks: EP2 x expert-DP2")
    rank = dist.get_rank()
    topology = {
        "world_size": 4, "dense_world": [0, 1, 2, 3],
        "expert_dp": [[0, 2], [1, 3]],
        "expert_parallel": [[0, 1], [2, 3]],
        "localization_collectives": ["dense_world", "expert_dp"],
        "excluded_collectives": ["expert_parallel"],
    }
    print(json.dumps({"rank": rank, "group": "dense_world", "members": [0, 1, 2, 3]}), flush=True)
    reports = [dict(name='dense_world', label='dense_DP', localization=True,
                    members=dist.get_process_group_ranks(dist.group.WORLD))]
    ep_group = dp_group = None
    # All ranks call new_group in exactly the same order, including nonmembers.
    for kind in ("expert_parallel", "expert_dp"):
        for members in topology[kind]:
            print(json.dumps({"rank": rank, "group": kind, "members": members,
                              "member": rank in members}), flush=True)
            group = dist.new_group(members)
            if rank in members:
                reports.append(dict(name=kind, label='Expert_DP' if kind == 'expert_dp' else None,
                                    localization=kind == 'expert_dp',
                                    members=dist.get_process_group_ranks(group)))
                if kind == "expert_parallel":
                    ep_group = group
                else:
                    dp_group = group
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(rank=rank, groups=reports))
    topology.update(build_topology(gathered, dist.get_world_size()))
    if rank == 0:
        (output / "topology.json").write_text(json.dumps(topology, indent=2))
    return ep_group, dp_group


def patch_block(module, ep_group):
    if not isinstance(module.experts, torch.nn.ModuleList):
        raise RuntimeError("This adapter requires transformers 4.53.x ModuleList Qwen2MoE experts")
    module._ep_size = dist.get_world_size(ep_group)
    if module.num_experts % module._ep_size:
        raise ValueError("Experts must be divisible by the actual EP group size")
    module._experts_per_rank = module.num_experts // module._ep_size
    module._ep_index = dist.get_rank(ep_group)
    module._ep_group = ep_group
    for expert_id in range(module.num_experts):
        if expert_id // module._experts_per_rank != module._ep_index:
            # Release nonowned parameters before moving the model to the GPU.
            module.experts[expert_id] = torch.nn.Identity()
    module.forward = types.MethodType(ep_forward, module)


def patch_moe(model, ep_group):
    count = 0
    for layer in model.model.layers:
        if layer.mlp.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
            patch_block(layer.mlp, ep_group)
            count += 1
    expected = sum(i not in model.config.mlp_only_layers and
                   (i + 1) % model.config.decoder_sparse_step == 0
                   for i in range(model.config.num_hidden_layers))
    if count != expected or count == 0:
        raise ValueError(f"Expected {expected} MoE layers, patched {count}")
    return count


def owned_parameter(name, experts_per_rank):
    match = re.search(r"(?:^|\.)experts\.(\d+)\.", name)
    if match is None:
        return None  # Shared experts, router, attention, embedding: dense world.
    return int(match.group(1)) // experts_per_rank == dist.get_rank() % 2


def synchronize_gradients(model, dp_group, experts_per_rank):
    """Stream fixed 4 MiB buckets; never flatten the full parameter set.

    Dense buckets have identical order across all ranks. Expert buckets have
    identical order across each DP pair, which owns the same expert indices.
    """
    dense, experts = [], []
    for name, parameter in model.named_parameters():
        owned = owned_parameter(name, experts_per_rank)
        if owned is False:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if not parameter.grad.is_contiguous():
            parameter.grad = parameter.grad.contiguous()
        (experts if owned else dense).append(parameter.grad.view(-1))
    for gradients, group in ((dense, None), (experts, dp_group)):
        if not gradients:
            continue
        capacity = 4 * 1024 * 1024 // gradients[0].element_size()
        bucket = gradients[0].new_empty(capacity)
        destinations, used = [], 0

        def flush():
            nonlocal destinations, used
            if not used:
                return
            dist.all_reduce(bucket[:used], group=group)
            # EP already sums experts' token gradients: divide by world, not DP.
            bucket[:used].div_(dist.get_world_size())
            offset = 0
            for destination in destinations:
                count = destination.numel()
                destination.copy_(bucket[offset:offset + count])
                offset += count
            destinations, used = [], 0

        for gradient in gradients:
            offset = 0
            while offset < gradient.numel():
                count = min(capacity - used, gradient.numel() - offset)
                destination = gradient[offset:offset + count]
                bucket[used:used + count].copy_(destination)
                destinations.append(destination)
                used += count
                offset += count
                if used == capacity:
                    flush()
        flush()


def verify_ep(ep_group, dp_group, device):
    """Check the exact HF block on a tiny config without loading full weights."""
    from transformers import Qwen2MoeConfig
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
    config = Qwen2MoeConfig(hidden_size=16, intermediate_size=24,
                           moe_intermediate_size=8, shared_expert_intermediate_size=12,
                           num_experts=4, num_experts_per_tok=2, norm_topk_prob=True)
    torch.manual_seed(2026)
    reference = Qwen2MoeSparseMoeBlock(config).to(device).float().train()
    sharded = copy.deepcopy(reference)
    patch_block(sharded, ep_group)
    # Include empty assignments: one local token can select at most two experts.
    for tokens in (1, 7):
        reference.zero_grad(set_to_none=True)
        sharded.zero_grad(set_to_none=True)
        torch.manual_seed(1000 + dist.get_rank() + tokens)
        x = torch.randn(1, tokens, 16, device=device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        actual, actual_router = sharded(x)
        expected, expected_router = reference(x_ref)
        (actual.square().mean() + actual_router.square().mean() * 0.01).backward()
        (expected.square().mean() + expected_router.square().mean() * 0.01).backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-6, rtol=2e-5)
        synchronize_gradients(sharded, dp_group, 2)
        actual_parameters = dict(sharded.named_parameters())
        for name, parameter in reference.named_parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(4)
            if name in actual_parameters:
                torch.testing.assert_close(actual_parameters[name].grad, parameter.grad,
                                           atol=2e-6, rtol=2e-5)
    dist.barrier()
    if dist.get_rank() == 0:
        print("EP verification passed: HF output/input/parameter gradients; empty assignments", flush=True)


class RedisWriter:
    def __init__(self, port, host="127.0.0.1"):
        self.connection = socket.create_connection((host, port), timeout=10)
        self.reader = self.connection.makefile("rb")

    def write(self, iteration, milliseconds):
        commands = [
            [b"RPUSH", b"iter_time_0", f"{iteration}:{milliseconds:.6f}".encode()],
            [b"LTRIM", b"iter_time_0", b"-500", b"-1"],
            [b"SET", b"latest_iteration", str(iteration).encode()],
        ]
        payload = b"".join(
            b"*%d\r\n" % len(parts) +
            b"".join(b"$%d\r\n" % len(p) + p + b"\r\n" for p in parts)
            for parts in commands
        )
        self.connection.sendall(payload)
        for index in range(len(commands)):
            reply = self.reader.readline()
            if not (reply.startswith(b":") if index == 0 else reply == b"+OK\r\n"):
                raise RuntimeError(f"Redis command failed: {reply!r}")


class GPUStreamDelay:
    """Explicit GPU stream delay simulation; no CPU sleep or collective hooks."""
    def __init__(self, args, device, output):
        self.args = args
        self.cycles = 0
        self.active = False
        self.log = None
        self.observed_calibration_ms = None
        if not args.compute_delay_ms or dist.get_rank() != args.fault_rank:
            return

        def measure(cycles):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            torch.cuda._sleep(cycles)
            end.record()
            end.synchronize()
            return begin.elapsed_time(end)

        samples = [measure(2_000_000) for _ in range(2)]
        measured = samples[1]
        if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
            raise RuntimeError(f"Invalid GPU stream delay calibration: {samples}")
        cycles = args.compute_delay_ms / measured * 2_000_000
        if not math.isfinite(cycles) or not 1 <= cycles < 2**63:
            raise RuntimeError(f"Invalid calibrated GPU delay cycles: {cycles}")
        self.cycles = int(round(cycles))
        observed = measure(self.cycles)
        if not math.isfinite(observed) or observed <= 0:
            raise RuntimeError(f"Invalid GPU stream delay validation: {observed}")
        if abs(observed - args.compute_delay_ms) > max(0.5, args.compute_delay_ms * 0.5):
            raise RuntimeError(f"GPU delay calibration mismatch: target={args.compute_delay_ms} ms, observed={observed} ms")
        self.observed_calibration_ms = observed
        self.log = (output / f"fault_rank{args.fault_rank}.jsonl").open("w", buffering=1)
        self.event(0, "calibration", calibration_samples_ms=samples,
                   target_ms=args.compute_delay_ms, observed_ms=observed)

    def event(self, iteration, kind, **extra):
        record = dict(iteration=iteration, type=kind,
                      simulation="simulate_gpu_stream_delay",
                      event=f"compute_delay_{kind}", kind="simulated_gpu_stream_delay",
                      delay_ms=self.args.compute_delay_ms,
                      backward_passes_per_step=getattr(self.args, 'microbatches', 1),
                      configured_total_delay_ms_per_step=(self.args.compute_delay_ms * getattr(self.args, 'microbatches', 1)),
                      observed_calibration_ms=self.observed_calibration_ms,
                      rank=self.args.fault_rank, start=self.args.delay_start,
                      end=self.args.delay_end, calibrated_cycles=self.cycles,
                      walltime=time.time(), **extra)
        self.log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def apply(self, iteration):
        if not self.cycles:
            return
        active = self.args.delay_start < iteration <= self.args.delay_end
        if active != self.active:
            self.event(iteration, "start" if active else "end")
            self.active = active
        if active:
            torch.cuda._sleep(self.cycles)

    def attach_backward(self, model):
        """Spread calibrated delay across actual decoder backward computation.

        Attention backward follows the layer's MoE backward. This avoids placing
        the fault in a collective or after overlapped reductions already finish.
        """
        self.iteration = 0
        self.hooks = []
        layers = model.model.layers
        modules = [layer.self_attn for layer in layers]
        experts = [expert for layer in layers if hasattr(layer.mlp, 'experts')
                   for expert in layer.mlp.experts if not isinstance(expert, torch.nn.Identity)]
        self.parts = len(modules)
        def inject(cycles):
            if self.cycles and self.args.delay_start < self.iteration <= self.args.delay_end:
                torch.cuda._sleep(max(1, round(cycles)))
        attention_cycles = self.cycles / len(modules)
        if self.args.delay_placement == 'attention-experts':
            if not experts:
                raise ValueError('attention-experts requires owned routed experts')
            attention_cycles /= 2
            expert_cycles = self.cycles / 2 / len(experts)
            for expert in experts:
                self.hooks.append(expert.register_full_backward_pre_hook(
                    lambda module, grad_output: inject(expert_cycles)))
            self.parts += len(experts)
        for attention in modules:
            self.hooks.append(attention.register_full_backward_pre_hook(
                lambda module, grad_output: inject(attention_cycles)))

    def begin_iteration(self, iteration):
        self.iteration = iteration
        if not self.cycles:
            return
        active = self.args.delay_start < iteration <= self.args.delay_end
        if active != self.active:
            self.event(iteration, 'start' if active else 'end',
                       placement=self.args.delay_placement, parts=self.parts)
            self.active = active

    def close(self, iteration):
        if self.log:
            if self.active:
                self.event(iteration, "end", reason="training_complete")
            self.log.close()


def load_model(path):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    config = Qwen2MoeConfig.from_pretrained(str(path), local_files_only=True)
    # The pinned snapshot was saved by v5.5; v4.53 reads rope_theta directly.
    rope = getattr(config, "rope_parameters", None)
    if rope:
        if rope.get("rope_type", "default") != "default":
            raise ValueError("This snapshot adapter expects default RoPE")
        config.rope_theta = rope["rope_theta"]
    model, info = Qwen2MoeForCausalLM.from_pretrained(
        str(path), config=config, local_files_only=True, torch_dtype=torch.bfloat16,
        output_loading_info=True, attn_implementation="sdpa",
    )
    missing = [k for k in info["missing_keys"]
               if k not in ("lm_head.weight", "model.embed_tokens.weight")]
    if missing or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete pretrained checkpoint: {info}")
    return model.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", default="models/qwen_moe")
    parser.add_argument("--output", default="flash_v1/output")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verify-ep", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compute-delay-ms", type=float, default=0.0)
    parser.add_argument("--delay-start", type=int, default=45)
    parser.add_argument("--delay-end", type=int, default=125)
    parser.add_argument("--fault-rank", type=int, default=3)
    parser.add_argument("--gradient-backend", choices=['serial', 'megatron'], default='serial')
    parser.add_argument("--bucket-size", type=int, default=2_097_152)
    parser.add_argument("--expert-parallel-size", type=int, default=2)
    parser.add_argument("--delay-placement", choices=['attention', 'attention-experts'], default='attention')
    args = parser.parse_args()
    if args.steps < 1 or args.seq_len < 2 or args.batch_size < 1 or args.warmup < 0:
        parser.error("steps/batch-size must be positive; seq-len >=2; warmup >=0")
    if not math.isfinite(args.compute_delay_ms) or args.compute_delay_ms < 0:
        parser.error("compute-delay-ms must be finite and nonnegative")
    if args.compute_delay_ms and args.device != "cuda":
        parser.error("GPU stream delay simulation requires --device cuda")
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if not 0 <= args.fault_rank < world_size or not 0 <= args.delay_start < args.delay_end:
        parser.error("fault-rank must be in world; require 0 <= delay-start < delay-end")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dense_group = None
    if args.gradient_backend == 'megatron':
        if device.type != 'cuda':
            parser.error('Original Megatron grad buffers require CUDA')
        from flash_v1 import megatron_adapter
        ep_group, dp_group, dense_group = megatron_adapter.initialize_groups(output, args.expert_parallel_size)
    else:
        ep_group, dp_group = create_topology(output)
    if device.type == "cuda":
        # Reserve NCCL's lazy communicator buffers before model/activation
        # allocations occupy GPU memory. This is outside every timed step.
        probe = torch.ones(1, device=device)
        for group in (None, ep_group, dp_group):
            dist.all_reduce(probe, group=group)
            torch.cuda.synchronize(device)
        del probe
    torch.manual_seed(2026)
    if args.verify_ep or args.verify_only:
        verify_ep(ep_group, dp_group, device)
    if args.verify_only:
        dist.destroy_process_group()
        return
    model = load_model(Path(args.model).resolve())
    total_parameters = sum(p.numel() for p in model.parameters())
    layers = patch_moe(model, ep_group)
    model = model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    delay = GPUStreamDelay(args, device, output)
    ddp = None
    if args.gradient_backend == 'megatron':
        ddp = megatron_adapter.wrap(model, dense_group, dp_group, args.bucket_size)
        delay.attach_backward(model)
        (output / f'buckets_rank{rank}.json').write_text(json.dumps(megatron_adapter.describe(ddp), indent=2))
    # Keep another rank's calibration time out of the first timed iteration.
    dist.barrier()
    experts_per_rank = model.config.num_experts // dist.get_world_size(ep_group)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, foreach=False)
    generator = torch.Generator(device=device).manual_seed(2026 + rank)
    metadata = dict(vars(args), model_id=MODEL_ID, revision=REVISION,
                    parameters=total_parameters,
                    local_parameters=sum(p.numel() for p in model.parameters()),
                    moe_layers=layers, corpus="synthetic_uniform_random_tokens",
                    fault_simulation="simulate_gpu_stream_delay" if args.compute_delay_ms else None,
                    task="systems_performance_only_not_language_quality")
    if rank == 0:
        (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        redis = RedisWriter(args.redis_port)
        log = (output / "train.jsonl").open("w", buffering=1)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    started = time.time()
    for iteration in range(1, args.warmup + args.steps + 1):
        batch = torch.randint(0, model.config.vocab_size,
                              (args.batch_size, args.seq_len + 1),
                              device=device, generator=generator)
        sync()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if ddp is not None:
            ddp.zero_grad_buffer()
            delay.begin_iteration(iteration)
        result = (ddp if ddp is not None else model)(input_ids=batch[:, :-1].contiguous(), use_cache=False,
                       output_router_logits=True)
        loss = F.cross_entropy(result.logits.float().reshape(-1, model.config.vocab_size), batch[:, 1:].reshape(-1))
        loss = loss + model.config.router_aux_loss_coef * result.aux_loss
        loss.backward()
        if ddp is not None:
            megatron_adapter.finish_gradients(ddp)
        else:
            delay.apply(iteration)
            synchronize_gradients(model, dp_group, experts_per_rank)
        update_rows = before_update = None
        if iteration == 1:
            # A few observed target-token rows prove an actual optimizer update
            # without retaining a second embedding table or full checkpoint.
            update_rows = torch.unique(batch[:, 1:].reshape(-1))[:8]
            before_update = model.get_input_embeddings().weight.detach()[update_rows].clone()
        optimizer.step()
        sync()
        elapsed_ms = (time.perf_counter() - begin) * 1000
        # Logging/Redis and loss measurement collectives are outside train timing.
        update_evidence = None
        if iteration == 1:
            after_update = model.get_input_embeddings().weight.detach()[update_rows]
            update_evidence = dict(
                parameter="model.embed_tokens.weight", rows=update_rows.tolist(),
                max_abs_delta=(after_update.float() - before_update.float()).abs().max().item(),
            )
        reduced_loss = loss.detach().float().clone()
        dist.all_reduce(reduced_loss)
        reduced_loss.div_(dist.get_world_size())
        if rank == 0:
            redis.write(iteration, elapsed_ms)
            record = dict(step=iteration, iteration=iteration, warmup=iteration <= args.warmup,
                          loss=reduced_loss.item(), time_ms=elapsed_ms,
                          declared_compute_delay_ms=(args.compute_delay_ms
                              if args.delay_start < iteration <= args.delay_end else 0.0),
                          walltime=time.time(), elapsed_walltime=time.time() - started)
            if update_evidence is not None:
                record["optimizer_update"] = update_evidence
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    delay.close(args.warmup + args.steps + 1)
    if rank == 0:
        log.close()
        redis.reader.close()
        redis.connection.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
