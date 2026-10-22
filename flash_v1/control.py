"""Single-node local/global controllers; Redis protocol also permits remote nodes."""
import argparse
import json
import mmap
from pathlib import Path
import struct
import time

import numpy as np
import redis
from flash_v1.analysis import WindowDetector, localize, profile_seconds
from flash_v1.topology import load_topology, label_cliques


def emit(path, event, **values):
    row = dict(event=event, wall_time=time.time(), **values)
    with path.open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def read_records(run_id, since_us, topology_path=None, diagnostics_path=None):
    rows = []
    topology, diagnostics = load_topology(topology_path)
    for path in sorted(Path('/dev/shm').glob(f'flash_record_{run_id}_*')):
        data = np.memmap(path, mode='r', dtype=np.uint64)
        for attempt in range(30):
            meta = data[:7].copy()
            fields, capacity, size, head = map(int, meta[:4])
            if fields != 14 or size > capacity:
                raise RuntimeError(f'invalid GREYHOUND ring: {path}')
            records = data[7:].reshape(capacity, fields)[(np.arange(size) + head) % capacity].copy()
            if np.array_equal(meta, data[:7]):
                break
        else:
            raise RuntimeError(f'GREYHOUND ring remained busy: {path}')
        for rec in records:
            if rec[11] == 0 or rec[7] < since_us:
                continue
            comm, kind, count, dtype, caller, clique, duration, n = map(int, rec[[0,1,2,5,9,10,11,12]])
            if kind == 6:
                rows.append(dict(rank=caller, comm=str(comm), clique=str(clique),
                    count=count, dtype=dtype, duration_us=duration, num_devices=n,
                    group_id=int(rec[3]), timestamp_us=int(rec[7]), buffer_addr=str(int(rec[4]))))
    rows, matching_diagnostics = label_cliques(rows, topology)
    diagnostics.extend(matching_diagnostics)
    if diagnostics_path is not None:
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    return rows


def local(args, client, output):
    state_path = Path(args.state_file)
    state_path.write_bytes(struct.pack('i', 0))
    with state_path.open('r+b') as state_file:
        state = mmap.mmap(state_file.fileno(), 4)
        active = False
        epoch = None
        client.set('local_ready', '1')
        while not client.exists('shutdown'):
            command = client.get('profile_epoch')
            requested = client.get('state') == '1'
            if requested and command != epoch:
                epoch = command
                since_us = time.monotonic_ns() // 1000
                state[:4] = struct.pack('i', 1)
                active = True
                emit(output / 'local.jsonl', 'profile_start', epoch=epoch, since_us=since_us)
                client.set(f'profile_started_{epoch}', '1')
            if active and not requested:
                state[:4] = struct.pack('i', 0)
                time.sleep(.5)  # Drain the last CUDA Event before aggregation.
                rows = read_records(args.run_id, since_us, output / 'topology.json',
                                    output / f'profile_{epoch}_diagnostics.json')
                (output / f'profile_{epoch}.json').write_text(json.dumps(rows))
                client.set(f'profile_result_{epoch}', json.dumps(rows))
                emit(output / 'local.jsonl', 'profile_end', epoch=epoch, samples=len(rows))
                active = False
            time.sleep(.1)
        state[:4] = struct.pack('i', 0)
        state.close()


def global_controller(args, client, output):
    detector = WindowDetector()
    reference_path = getattr(args, 'reference_profile', None)
    if reference_path:
        raise ValueError('Healthy PROFILE differential localization is temporarily disabled')
    baseline_rows = json.loads(Path(reference_path).read_text()) if reference_path else None
    last_iteration, deadline, epoch = 0, None, 0
    while not client.exists('shutdown'):
        # The producer trims this list to 500 items; a positional cursor would
        # stop observing new steps once the list reaches that length.
        for encoded in client.lrange('iter_time_0', 0, -1):
            iteration, duration = encoded.split(':')
            iteration = int(iteration)
            if iteration <= last_iteration:
                continue
            last_iteration = iteration
            if deadline is not None:
                continue  # Profiling perturbs step time; it cannot be monitoring evidence.
            event = detector.add(int(iteration), float(duration))
            manual_at = client.get('manual_profile_iteration')
            manual = manual_at is not None and iteration >= int(manual_at) and epoch == 0
            if manual:
                event = dict(iteration=iteration, profile_seconds=profile_seconds(float(duration)))
            if event is not None:
                epoch += 1
                emit(output / 'events.jsonl', 'manual_profile' if manual else 'detection', **event)
                client.set('profile_epoch', epoch)
                client.set('state', '1')
                # Begin the dynamic profiling window after the local probe is
                # enabled, so Redis polling delay cannot shorten the sample.
                limit = time.monotonic() + 10
                while not client.exists(f'profile_started_{epoch}') and time.monotonic() < limit:
                    if client.exists('shutdown'):
                        client.set('state', '0')
                        return
                    time.sleep(.1)
                if not client.exists(f'profile_started_{epoch}'):
                    raise TimeoutError('local controller profile start missing')
                deadline = time.monotonic() + profile_seconds(float(duration))
        if deadline is not None and time.monotonic() >= deadline:
            client.set('state', '0')
            deadline = None
            key = f'profile_result_{epoch}'
            limit = time.monotonic() + 10
            while not client.exists(key) and time.monotonic() < limit:
                time.sleep(.1)
            if not client.exists(key):
                raise TimeoutError('local controller profile result missing')
            rows = json.loads(client.get(key))
            topology, _ = load_topology(output / 'topology.json')
            options = {}
            if topology:
                options['topology'] = topology
            if baseline_rows is not None:
                options['baseline_rows'] = baseline_rows
            result = localize(rows, **options) if rows else dict(status='inconclusive', rank=None, reason='no probe samples')
            emit(output / 'events.jsonl', 'localization', epoch=epoch, result=result)
            # Turning profiling off precedes aggregation. The runner must wait
            # for this explicit result, rather than treating state=0 as done.
            client.set(f'profile_completed_{epoch}', '1')
            # Fresh healthy/degraded baseline after a profile; no stale-change rediscovery.
            # Also discard steps completed while awaiting local aggregation.
            # The first subsequent complete monitoring step supplies a fresh baseline.
            pending = client.lrange('iter_time_0', -1, -1)
            if pending:
                last_iteration = max(last_iteration, int(pending[0].split(':')[0]))
            detector = WindowDetector(warmup=last_iteration)
        client.set('processed_iteration', last_iteration)
        time.sleep(.1)
    client.set('state', '0')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['local', 'global'])
    p.add_argument('--redis-port', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--state-file', required=True)
    p.add_argument('--reference-profile')
    args = p.parse_args()
    client = redis.Redis(port=args.redis_port, decode_responses=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (local if args.role == 'local' else global_controller)(args, client, output)

if __name__ == '__main__':
    main()
