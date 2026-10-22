"""Complete pretrained Qwen MoE on original Megatron PP2/EP2/ExpertDP2."""
import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from flash_v1 import megatron_adapter
from flash_v1.collection import CollectionSession, add_arguments
from flash_v1.pipeline_adapter import StageModel, forward_step
from flash_v1.train import GPUStreamDelay, RedisWriter, load_model, patch_moe, MODEL_ID, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seq-len', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--microbatches', type=int, default=2)
    p.add_argument('--model', default='models/qwen_moe')
    p.add_argument('--output', required=True)
    p.add_argument('--redis-host', default='127.0.0.1')
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--redis-port', type=int, default=16422)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bucket-size', type=int, default=2_097_152)
    p.add_argument('--expert-parallel-size', type=int, default=2)
    p.add_argument('--gradient-backend', default='pipeline')
    p.add_argument('--compute-delay-ms', type=float, default=0)
    p.add_argument('--fault-rank', type=int, default=7)
    p.add_argument('--delay-start', type=int, default=45)
    p.add_argument('--delay-end', type=int, default=125)
    p.add_argument('--delay-placement', choices=['attention', 'attention-experts'], default='attention-experts')
    p.add_argument('--online', action='store_true', help='Automatic detection and synchronized bounded profiling')
    p.add_argument('--monitor-benchmark', action='store_true', help='Controlled, zero-learning-rate MONITOR crossover measurement')
    add_arguments(p)
    args = p.parse_args()
    if args.collection_plan:
        plan = json.loads(Path(args.collection_plan).read_text())
        intervals = plan if isinstance(plan, list) else plan['intervals']
        args.steps = max(interval['end'] for interval in intervals)
        args.warmup = 0
    if args.monitor_benchmark:
        from flash_v1.scripts.monitor16_crossover import benchmark_blocks, benchmark_step
        if (not args.online or not args.collection_plan or not isinstance(plan, dict)
                or not plan.get('monitor_benchmark') or plan.get('benchmark_blocks') != benchmark_blocks()
                or args.steps != 268 or any(row.get('target_ranks') or row.get('delay_ms', 0) for row in intervals)):
            p.error('monitor benchmark requires the declared 268-step healthy online crossover plan')
        args.lr = 0.0
    if args.tensor_parallel_size not in (1, 2) or args.expert_parallel_size != 2 or args.steps < 1 or args.microbatches < 1:
        p.error('PP2 uses EP2; steps and microbatches must be positive')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    if dist.get_world_size() != 8 * args.tensor_parallel_size:
        raise ValueError('This PP2×EP2×ExpertDP2 experiment requires 8*TP GPUs')
    rank = dist.get_rank()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ep, expert_dp, dense = megatron_adapter.initialize_groups(output, 2, 2, args.tensor_parallel_size)
    state, _, _ = megatron_adapter.core()
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    # Allocate actual collective buffers outside timed training.
    probe = torch.ones(1, device='cuda')
    for group in (dense, ep, expert_dp, state.get_embedding_group()):
        dist.all_reduce(probe, group=group)
        torch.cuda.synchronize()
    del probe
    torch.manual_seed(2026)
    full = load_model(Path(args.model).resolve())
    total_parameters = sum(p.numel() for p in full.parameters())
    patch_moe(full, ep)
    stage = StageModel(full, state.get_pipeline_model_parallel_rank())
    from flash_v1.tensor_parallel import patch_tensor_parallel
    tp_manifest = patch_tensor_parallel(stage, state.get_tensor_model_parallel_group())
    del full
    stage.cuda()
    ddp = megatron_adapter.wrap(stage, dense, expert_dp, args.bucket_size, config=stage.config)
    stage.config.no_sync_func = ddp.no_sync
    stage.config.finalize_model_grads_func = finalize_model_grads
    delay = (CollectionSession(args, output) if args.collection_plan else
             GPUStreamDelay(args, torch.device('cuda'), output))
    delay.attach_backward(SimpleNamespace(model=stage.decoder))
    optimizer = torch.optim.SGD(stage.parameters(), lr=args.lr, foreach=False)
    generator = torch.Generator(device='cuda').manual_seed(args.seed + state.get_data_parallel_rank())
    benchmark_batches = []
    if args.monitor_benchmark:
        for _ in range(32):
            batches = []
            for _ in range(args.microbatches):
                tokens = torch.randint(0, stage.hf_config.vocab_size,
                    (args.batch_size, args.seq_len + 1), device='cuda', generator=generator)
                batches.append(dict(input_ids=tokens[:, :-1].contiguous(), next_token_labels=tokens[:, 1:].contiguous()))
            benchmark_batches.append(batches)
    (output / f'buckets_rank{rank}.json').write_text(json.dumps(megatron_adapter.describe(ddp), indent=2))
    if rank == 0:
        (output / 'run.json').write_text(json.dumps(dict(vars(args), model_id=MODEL_ID,
            revision=REVISION, parameters=total_parameters, pipeline_stages=2,
            tensor_parallel=tp_manifest,
            task='systems_performance_only_not_language_quality',
            pipeline_payload='FP32 hidden plus differentiable router sufficient statistics',
            batch_p2p_sync=stage.config.batch_p2p_sync), indent=2))
        # The collection/online harness has its own CPU consumer. Legacy
        # single-node controllers retain the original Redis metric protocol.
        writer = None if args.collection_plan else RedisWriter(args.redis_port, host=args.redis_host)
        log = (output / 'train.jsonl').open('w', buffering=1)
    current_epoch = 0
    probe_enabled = False
    have_control = False
    if args.online:
        if not args.collection_plan:
            p.error('--online currently requires the 16-GPU collection harness')
        probe_file = Path(os.environ['FLASH_STATE_FILE']).open('r+b')
        probe_state = mmap.mmap(probe_file.fileno(), 4)
        metric_buffer = torch.zeros(2, device='cuda')
        metric_cpu = torch.empty(2, pin_memory=True)
        if rank == 0:
            from flash_v1.online import OnlineSchedule
            from flash_v1.monitor_channel import MonitorChannel
            monitor_channel = MonitorChannel(output / 'monitor.bin')
            online_schedule = OnlineSchedule(None, output, channel=monitor_channel)
    dist.barrier()
    started = time.time()
    completed_perf = time.perf_counter()
    schedule = get_forward_backward_func()
    for iteration in range(1, args.warmup + args.steps + 1):
        benchmark = benchmark_step(iteration) if args.monitor_benchmark else None
        online_step = args.online and (benchmark is None or benchmark['monitor_active'])
        if benchmark is not None:
            batches = benchmark_batches[benchmark['benchmark_position'] - 1]
            current_epoch = 0
        else:
            batches = []
            for _ in range(args.microbatches):
                tokens = torch.randint(0, stage.hf_config.vocab_size,
                    (args.batch_size, args.seq_len + 1), device='cuda', generator=generator)
                batches.append(dict(input_ids=tokens[:, :-1].contiguous(), next_token_labels=tokens[:, 1:].contiguous()))
        torch.cuda.synchronize()
        if online_step and have_control:
            # Previous asynchronous D2H is complete at this existing boundary.
            # No extra host synchronization on the other fifteen ranks.
            current_epoch = round(float(metric_cpu[1]))
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        ddp.zero_grad_buffer()
        if args.online:
            # The preceding existing WORLD metric collective distributes one
            # common epoch. All GPUs finished its previous training step first.
            enabled = current_epoch > 0
            if enabled != probe_enabled:
                probe_state[:] = struct.pack('i', int(enabled))
                delay.set_profile_enabled(enabled)
                probe_enabled = enabled
        delay.begin_iteration(iteration)
        if iteration == 1:
            # The tied vocabulary weight owns local rows, not global token IDs.
            local_vocab = stage.shared_embedding_or_output_weight().shape[0]
            vocab_start = state.get_tensor_model_parallel_rank() * local_vocab
            token_rows = torch.unique(batches[0]['next_token_labels']) - vocab_start
            update_rows = token_rows[(token_rows >= 0) & (token_rows < local_vocab)][:8]
            if update_rows.numel() == 0:
                update_rows = torch.arange(min(8, local_vocab), device='cuda')
            before = stage.shared_embedding_or_output_weight().detach()[update_rows].clone()
        metrics = schedule(forward_step_func=forward_step, data_iterator=iter(batches),
            model=ddp, num_microbatches=args.microbatches,
            seq_length=args.seq_len + stage.extra_positions(args.batch_size),
            micro_batch_size=args.batch_size)
        # The schedule invokes the original finalizer, including tied PP grads.
        for parameter in stage.parameters():
            parameter.grad = parameter.main_grad
        optimizer.step()
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - begin) * 1000
        if args.collection_plan:
            delay.end_iteration(iteration, milliseconds)
        value = metric_buffer.zero_() if online_step else torch.zeros((), device='cuda')
        if metrics:
            loss = torch.stack([x['loss'].float() for x in metrics]).mean()
            if online_step:
                value[0].copy_(loss)
            else:
                value = loss
        if online_step and rank == 0:
            next_epoch = online_schedule.next_epoch(iteration)
            if args.monitor_benchmark and next_epoch:
                raise RuntimeError('controlled healthy benchmark received a PROFILE request')
            # CUDA Tensor.__setitem__(Python scalar) makes a synchronous H2D
            # copy in this Torch version. zero_ already set the MONITOR slot;
            # nonzero commands use an asynchronous GPU fill kernel instead.
            if next_epoch:
                value[1].fill_(next_epoch)
        dist.all_reduce(value)
        if online_step:
            # Same WORLD control protocol; stage the result asynchronously.
            metric_cpu.copy_(value, non_blocking=True)
            have_control = True
        else:
            value.div_(dist.get_world_size(dense) * args.tensor_parallel_size)
            if args.monitor_benchmark:
                have_control = False
        if rank == 0:
            if online_step:
                # One D2H/sync replaces two scalar reads and the GPU loss-divide
                # kernel. This still includes completion of the WORLD metric.
                torch.cuda.current_stream().synchronize()
                loss_scalar = metric_cpu.tolist()[0] / (dist.get_world_size(dense) * args.tensor_parallel_size)
            else:
                loss_scalar = value.item()
            now_perf = time.perf_counter()
            wall_iteration_ms = (now_perf - completed_perf) * 1000
            completed_perf = now_perf
            record = dict(step=iteration, iteration=iteration, warmup=iteration <= args.warmup,
                          loss=loss_scalar, time_ms=milliseconds, walltime=time.time(),
                          wall_iteration_ms=wall_iteration_ms, profile_epoch=current_epoch,
                          declared_compute_delay_ms=args.compute_delay_ms * args.microbatches if args.delay_start < iteration <= args.delay_end else 0,
                          elapsed_walltime=time.time() - started)
            if benchmark is not None:
                record.update(benchmark, warmup=iteration <= 12)
            if iteration == 1:
                after = stage.shared_embedding_or_output_weight().detach()[update_rows]
                record['optimizer_update'] = dict(parameter='tied_embedding', rows=update_rows.tolist(),
                    max_abs_delta=(after.float() - before.float()).abs().max().item())
            if args.online:
                monitor_channel.publish(iteration, wall_iteration_ms, current_epoch, record['walltime'])
            if writer is not None:
                writer.write(iteration, milliseconds)
            log.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    delay.close(args.warmup + args.steps + 1)
    if rank == 0:
        log.close()
        if writer is not None:
            writer.reader.close()
            writer.connection.close()
        if args.online:
            monitor_channel.close()
    if args.online:
        probe_state[:] = struct.pack('i', 0)
        probe_state.close(); probe_file.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
