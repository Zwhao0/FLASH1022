"""Complete pretrained Qwen MoE on original Megatron PP2/EP2/ExpertDP2."""
import argparse
from contextlib import nullcontext
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
from flash_v1.train import GPUStreamDelay, RedisWriter, load_model, patch_moe, MODEL_ID, REVISION, stage_forward_observer


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
    evidence = None
    if args.collection_plan and plan.get('evidence_enabled', False):
        from flash_v1.evidence import EvidenceCollector
        evidence = EvidenceCollector(torch, stage.decoder, output, rank,
                                     selection_index=plan.get('evidence_selection_index', 0))
    stage_aux = None
    sparse_aux_enabled = bool(args.collection_plan and plan.get('stage_forward_aux_sampling') is not None)
    if sparse_aux_enabled:
        from flash_v1.stage_forward_sampling import CONFIG
        if (plan['stage_forward_aux_sampling'] != CONFIG or not args.online
                or not plan.get('stage_forward_aux_enabled', False)):
            raise ValueError('sparse3 requires explicit online auxiliary sampling contract')
    auxiliary_benchmark_blocks = []
    if args.collection_plan and plan.get('stage_forward_monitor_benchmark_blocks'):
        from flash_v1.stage_forward_benchmark import validate_blocks
        auxiliary_benchmark_blocks = validate_blocks(plan['stage_forward_monitor_benchmark_blocks'])
    if args.collection_plan and plan.get('stage_forward_aux_enabled', False):
        if args.microbatches != 2 or args.tensor_parallel_size != 2 or evidence is None:
            raise ValueError('stage FWD auxiliary requires the attested TP2/two-MB evidence workload')
        from megatron.core.pipeline_parallel import schedules as native_schedules
        from flash_v1.stage_forward_evidence import StageForwardEvidence
        if sparse_aux_enabled:
            from flash_v1.stage_forward_sparse_collector import StageForwardEvidence
        layers = list(stage.decoder.layers)
        mlps = sum(type(module).__name__ == 'Qwen2MoeMLP' for module in stage.modules())
        ep_layers = [layer for layer in layers if hasattr(layer.mlp, '_ep_group')]
        stage_aux = StageForwardEvidence(torch, stage, native_schedules, dist, rank,
            expected_collectives=dict(all_reduce=len(layers) + mlps + int(stage.pre_process),
                all_to_all_single=4 * len(ep_layers), all_gather=int(stage.post_process)),
            metadata_spans=len(ep_layers),
            stage_metadata=dict(layout=plan.get('parallel_layout', 'legacy'),
                pp_stage=state.get_pipeline_model_parallel_rank(), tp_size=args.tensor_parallel_size,
                tp_group_ranks=dist.get_process_group_ranks(state.get_tensor_model_parallel_group()),
                model_id=MODEL_ID, model_revision=REVISION, compute_dtype=str(stage.parameter_dtype),
                layer_indices=[layer.self_attn.layer_idx for layer in layers],
                ep_layer_indices=[layer.self_attn.layer_idx for layer in ep_layers],
                dense_layer_indices=[layer.self_attn.layer_idx for layer in layers if layer not in ep_layers],
                hidden_size=stage.hf_config.hidden_size, pre_process=stage.pre_process,
                post_process=stage.post_process))
    whole = None
    whole_truth = []
    scoped_compute = bool(args.collection_plan and plan.get('scoped_compute_enabled', False))
    if args.collection_plan and (any(row.get('fault_mode') == 'whole_compute' for row in intervals)
                                 or plan.get('scoped_catalog_only', False)):
        if scoped_compute:
            from flash_v1.scoped_compute import ScopedComputeSlowdown
            whole = ScopedComputeSlowdown(torch, rank=rank)
            aliases = {}
            for name, module in stage.named_modules(remove_duplicate=False):
                aliases.setdefault(id(module), []).append(name or '<model>')
            (output / f'scoped_aliases_rank{rank}.json').write_text(json.dumps(dict(
                aliases=[names for names in aliases.values() if len(names)>1],
                slot_policy='canonical module object/phase/operator ordinal; repeated invocations and Python aliases share slots'), indent=2))
            whole.attach(stage)
        else:
            from flash_v1.whole_compute import WholeComputeSlowdown
            whole = WholeComputeSlowdown(torch, rank=rank)
        whole_catalog = output / f'whole_compute_calibration_rank{rank}.json'
        reference = plan.get('whole_compute_reference_session')
        if reference:
            if Path(reference).name != reference:
                raise ValueError('whole compute reference must be a session name')
            whole.load_calibration(output.parent / reference / whole_catalog.name, whole_catalog)
        if scoped_compute and plan.get('scoped_reference_session'):
            scope_reference = plan['scoped_reference_session']
            if Path(scope_reference).name != scope_reference:
                raise ValueError('scoped reference must be a session name')
            whole.load_scope_catalog(output.parent / scope_reference / f'scope_catalog_rank{rank}.json')
    if scoped_compute and whole is None:
        raise ValueError('scoped compute requires catalog collection or scoped fault intervals')
    if stage_aux is not None:
        from flash_v1.stage_forward_context import observed_context
        stage_aux.metadata['instrumentation_context'] = observed_context(whole)
        (output / f'stage_forward_manifest_rank{rank}.json').write_text(json.dumps(dict(
            **({'sampling_contract': plan['stage_forward_aux_sampling']} if sparse_aux_enabled else {}),
            expected_collectives=stage_aux.expected, ep_metadata_spans=stage_aux.metadata_spans,
            mlp_keys=sorted(stage_aux.modules), stage_metadata=stage_aux.metadata,
            observer_class=type(whole).__name__ if whole is not None else None,
            scoped_hooks_installed=bool(getattr(whole, 'handles', []))), indent=2))
    if args.collection_plan and plan.get('benchmark_work_lock', False):
        from flash_v1.scripts.v2_benchmark import lock_workload
        lock_workload(output, rank, delay, whole,
                      plan.get('benchmark_work_reference_session'))
    native_fma = None
    if args.collection_plan and plan.get('native_module_fma_cases'):
        if whole is not None or scoped_compute:
            raise ValueError('native FMA integration requires plain no-dispatch training')
        from flash_v1.native_module_fma import NativeModuleFMA
        slope_path=Path(plan['native_fma_reference_files'][str(rank)])
        reference=json.loads(slope_path.read_text())
        if reference['rank']!=rank:
            raise ValueError('native FMA reference rank mismatch')
        native_fma=NativeModuleFMA(torch,stage,rank,plan['native_module_fma_cases'],
                                   reference['fma_model']['units_per_ms'])
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
        metric_zero = torch.zeros((), device='cuda')
        # Only rank zero consumes loss; the other ranks need one epoch scalar.
        # Cache views once rather than create a CUDA view on every iteration.
        metric_source = metric_buffer if rank == 0 else metric_buffer[1:]
        metric_cpu = torch.empty(2 if rank == 0 else 1, pin_memory=True)
        if rank == 0:
            from flash_v1.online import OnlineSchedule
            from flash_v1.monitor_channel import MonitorChannel
            monitor_channel = MonitorChannel(output / 'monitor.bin')
            online_schedule = OnlineSchedule(None, output, channel=monitor_channel)
    cost_audit = bool(args.collection_plan and plan.get('monitor_cost_timing', False))
    cost_rows = []
    startup_heap = None
    if args.collection_plan and ('startup_gc_fixed_model' in plan or plan.get('startup_gc_policy') is not None):
        from flash_v1.startup_gc import install
        startup_heap = install(torch, rank, plan, output)
    startup_status = 'aborted'
    try:
        dist.barrier()
        started = time.time()
        completed_perf = time.perf_counter()
        schedule = get_forward_backward_func()
        for iteration in range(1, args.warmup + args.steps + 1):
            if whole is not None and not reference:
                if iteration == 4:
                    whole.begin_calibration()
                elif iteration == 7:
                    whole.finish_calibration(whole_catalog)
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
            if cost_audit:
                cost_begin = time.perf_counter()
            torch.cuda.synchronize()
            if cost_audit:
                cost_ready = time.perf_counter()
            if online_step and have_control:
                # Previous asynchronous D2H is complete at this existing boundary.
                # No extra host synchronization on the other fifteen ranks.
                current_epoch = round(float(metric_cpu[-1]))
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            ddp.zero_grad_buffer()
            if args.online:
                # The preceding existing WORLD metric collective distributes one
                # common epoch. All GPUs finished its previous training step first.
                enabled = current_epoch > 0
                if enabled != probe_enabled:
                    native_enabled = enabled and plan.get('probe_enabled', True)
                    probe_state[:] = struct.pack('i', int(native_enabled))
                    delay.set_profile_enabled(native_enabled)
                    probe_enabled = enabled
            delay.begin_iteration(iteration)
            if whole is not None:
                row = delay.interval
                ratio = (row.get('slowdown_extra_ratio', 0)
                         if rank in row.get('target_ranks', []) else 0)
                if scoped_compute:
                    whole.begin_iteration(iteration, extra_ratio=ratio,
                                          target=row.get('fault_scope'), case_id=row.get('case_id'))
                else:
                    whole.begin_iteration(iteration, extra_ratio=ratio)
            if iteration == 1:
                # The tied vocabulary weight owns local rows, not global token IDs.
                local_vocab = stage.shared_embedding_or_output_weight().shape[0]
                vocab_start = state.get_tensor_model_parallel_rank() * local_vocab
                token_rows = torch.unique(batches[0]['next_token_labels']) - vocab_start
                update_rows = token_rows[(token_rows >= 0) & (token_rows < local_vocab)][:8]
                if update_rows.numel() == 0:
                    update_rows = torch.arange(min(8, local_vocab), device='cuda')
                before = stage.shared_embedding_or_output_weight().detach()[update_rows].clone()
            if native_fma is not None:
                native_fma.begin_iteration(iteration)
            active_stage_aux = stage_aux
            if auxiliary_benchmark_blocks:
                from flash_v1.stage_forward_benchmark import active_auxiliary
                active_stage_aux = active_auxiliary(stage_aux, auxiliary_benchmark_blocks, iteration, current_epoch)
            with (native_fma.loss_wrapper(torch.nn.functional) if native_fma is not None else nullcontext()), (whole if whole is not None else nullcontext()):
                evidence_active = (current_epoch > 0 if args.online else
                                   bool(plan.get('profile_enabled', True)) if args.collection_plan else False)
                if evidence_active and sparse_aux_enabled and active_stage_aux is not None and not active_stage_aux.selected(iteration, True, current_epoch if args.online else 1):
                    active_stage_aux = None
                with (stage_forward_observer(active_stage_aux if evidence_active else None)
                      if active_stage_aux is not None else nullcontext()), (active_stage_aux.iteration(iteration, enabled=evidence_active,
                                         epoch=current_epoch if args.online else 1)
                      if active_stage_aux is not None else nullcontext()), (evidence.iteration(iteration, enabled=evidence_active,
                                         epoch=current_epoch if args.online else 1)
                      if evidence is not None else nullcontext()):
                    if evidence_active and sparse_aux_enabled and active_stage_aux is not None:
                        active_stage_aux.pending['sampling_contract'] = active_stage_aux.sampler.manifest(current_epoch if args.online else 1)
                    with (whole.training() if scoped_compute else nullcontext()):
                        metrics = schedule(forward_step_func=forward_step, data_iterator=iter(batches),
                            model=ddp, num_microbatches=args.microbatches,
                            seq_length=args.seq_len + stage.extra_positions(args.batch_size),
                            micro_batch_size=args.batch_size)
                    # The schedule invokes the original finalizer, including tied PP grads.
                    for parameter in stage.parameters():
                        parameter.grad = parameter.main_grad
                    if args.collection_plan and iteration in plan.get('numerical_audit_steps', []):
                        from flash_v1.v2_numeric import record_gradients
                        record_gradients(torch, stage, output, rank, iteration)
                    optimizer.step()
                    if args.collection_plan and iteration in plan.get('stage_forward_numerical_audit_steps', []):
                        from flash_v1.stage_forward_numeric import record_parameters
                        record_parameters(torch, stage, output, rank, iteration)
            torch.cuda.synchronize()
            if evidence is not None:
                evidence.drain()
            if active_stage_aux is not None:
                auxiliary = active_stage_aux.drain()
                if auxiliary is not None:
                    with (output / f'stage_forward_aux_rank{rank}.jsonl').open('a') as stream:
                        stream.write(json.dumps(auxiliary) + '\n')
            milliseconds = (time.perf_counter() - begin) * 1000
            if args.collection_plan:
                delay.end_iteration(iteration, milliseconds)
            if native_fma is not None:
                native_fma.end_iteration()
            if whole is not None and (ratio or scoped_compute):
                # Keep detailed experiment truth off the timed training/monitor path.
                # Each rank owns one fault episode; serialize only after training.
                whole_truth.append(whole.end_iteration())
            if cost_audit:
                cost_train_end = time.perf_counter()
            value = metric_buffer if online_step else torch.zeros((), device='cuda')
            if metrics:
                loss = torch.stack([x['loss'].float() for x in metrics]).mean()
                if online_step:
                    # One stack kernel supplies loss and zero command, replacing
                    # a full zero plus a separate device copy on the last PP stage.
                    torch.stack((loss, metric_zero), out=value)
                else:
                    value = loss
            elif online_step:
                value.zero_()
            if online_step and rank == 0:
                next_epoch = online_schedule.next_epoch(iteration)
                if args.monitor_benchmark and next_epoch:
                    raise RuntimeError('controlled healthy benchmark received a PROFILE request')
                # CUDA Tensor.__setitem__(Python scalar) makes a synchronous H2D
                # copy in this Torch version. zero_ already set the MONITOR slot;
                # nonzero commands use an asynchronous GPU fill kernel instead.
                if next_epoch:
                    value[1].fill_(next_epoch)
            if cost_audit:
                cost_prepared = time.perf_counter()
            dist.all_reduce(value)
            if cost_audit:
                cost_reduced = time.perf_counter()
            if online_step:
                # Same WORLD control protocol; stage the result asynchronously.
                metric_cpu.copy_(metric_source, non_blocking=True)
                have_control = True
            else:
                value.div_(dist.get_world_size(dense) * args.tensor_parallel_size)
                if args.monitor_benchmark:
                    have_control = False
            if cost_audit:
                cost_staged = time.perf_counter()
            if rank == 0:
                if online_step:
                    # One D2H/sync replaces two scalar reads and the GPU loss-divide
                    # kernel. This still includes completion of the WORLD metric.
                    torch.cuda.current_stream().synchronize()
                    loss_scalar = metric_cpu.tolist()[0] / (dist.get_world_size(dense) * args.tensor_parallel_size)
                else:
                    loss_scalar = value.item()
                if cost_audit:
                    cost_scalar = time.perf_counter()
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
                if cost_audit:
                    cost_published = time.perf_counter()
                log.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            if cost_audit:
                cost_done = time.perf_counter()
                cost_rows.append(dict(iteration=iteration, epoch=current_epoch,
                    initial_sync_ms=1000*(cost_ready-cost_begin),
                    training_ms=1000*(cost_train_end-cost_ready),
                    metric_prepare_ms=1000*(cost_prepared-cost_train_end),
                    collective_launch_ms=1000*(cost_reduced-cost_prepared),
                    stage_copy_ms=1000*(cost_staged-cost_reduced),
                    scalar_wait_ms=1000*(cost_scalar-cost_staged) if rank == 0 else None,
                    publication_ms=1000*(cost_published-cost_scalar) if rank == 0 else None,
                    logging_ms=1000*(cost_done-cost_published) if rank == 0 else None))
        startup_status = 'complete'
    finally:
        if startup_heap is not None:
            from flash_v1.startup_gc import restore_owned
            restore_owned(startup_heap, output, startup_status)
    if cost_audit:
        (output / f'monitor_cost_rank{rank}.json').write_text(json.dumps(cost_rows))
    delay.close(args.warmup + args.steps + 1)
    if whole is not None:
        with (output / f'whole_compute_truth_rank{rank}.jsonl').open('w') as truth:
            for row in whole_truth:
                truth.write(json.dumps(row) + '\n')
        if scoped_compute:
            (output / f'scope_catalog_raw_rank{rank}.json').write_text(
                json.dumps(whole.scope_catalog(), indent=2))
            whole.detach()
    if native_fma is not None:
        with (output / f'native_module_fma_truth_rank{rank}.jsonl').open('w') as truth:
            for row in native_fma.stats:
                truth.write(json.dumps(row)+'\n')
        native_fma.close()
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
