"""Persist fresh ring records under the producer mutex; no GPU API calls."""
import argparse
import ctypes
import gzip
import json
import mmap
from pathlib import Path
import signal
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--redis-host', help='Publish bounded timed rows for the online controller')
    p.add_argument('--redis-port', type=int, default=17422)
    p.add_argument('--node-rank', type=int)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    library = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / 'build/libflashsnapshot.so'))
    snapshot = library.flash_snapshot
    pointer = ctypes.POINTER(ctypes.c_uint64)
    snapshot.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64,
                         pointer, ctypes.c_size_t, pointer]
    snapshot.restype = ctypes.c_int64
    buffer = np.empty((20_000, 14), dtype=np.uint64)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cursor, dropped, initializing = {}, [], {}
    samples = 0
    state_file = (output / 'state.bin').open('rb')
    probe_state = mmap.mmap(state_file.fileno(), 4, access=mmap.ACCESS_READ)
    client = None
    if args.redis_host:
        import redis
        client = redis.Redis(host=args.redis_host, port=args.redis_port,
                             socket_timeout=5, socket_connect_timeout=5)
        for attempt in range(100):
            try:
                client.ping()
                break
            except redis.ConnectionError:
                if attempt == 99:
                    raise
                time.sleep(.1)
    with gzip.open(output / 'raw_profile.jsonl.gz', 'wt', compresslevel=1) as stream:
        while True:
            if (client is not None and not int.from_bytes(probe_state[:4], 'little')
                    and not int(client.get('online_pending') or 0)
                    and not stopping and not (output / 'drain.stop').exists()):
                # No records are produced in MONITOR. A pending window keeps
                # draining even after the eight-step probe switch turns off.
                time.sleep(.5)
                continue
            published = []
            loss_start = len(dropped)
            # Read BEFORE copying rings. WORLD completion at this iteration
            # precedes all the snapshots below; no cross-node clock comparison.
            through_iteration = int(client.get('online_latest_iteration') or 0) if client is not None else 0
            for path in sorted(Path('/dev/shm').glob(f'flash_record_{args.run_id}_*')):
                last = cursor.get(str(path), -1)
                missed = ctypes.c_uint64()
                lock_name = path.name.replace('flash_record_', 'flash_lock_', 1)
                count = snapshot(path.name.encode(), lock_name.encode(), last,
                    buffer.ctypes.data_as(pointer), len(buffer), ctypes.byref(missed))
                if count == -1:
                    initializing.setdefault(str(path), time.monotonic())
                    if time.monotonic() - initializing[str(path)] > 10:
                        raise RuntimeError(f'ring never initialized {path}')
                    continue
                if count < 0:
                    raise RuntimeError(f'snapshot failed code={count}: {path}')
                initializing.pop(str(path), None)
                if missed.value:
                    dropped.append(dict(file=path.name, after=last, missed=int(missed.value)))
                if count:
                    cursor[str(path)] = int(buffer[count - 1, 13])
                for rec in buffer[:count]:
                    if int(rec[11]) <= 0 or int(rec[4]) <= 0:
                        continue
                    row = dict(rank=int(rec[9]), comm=str(int(rec[0])), kind=int(rec[1]),
                        clique=str(int(rec[10])), count=int(rec[2]), dtype=int(rec[5]),
                        group_id=int(rec[3]), iteration=int(rec[4]),
                        operation_index=int(rec[6]) & 0xffffffff, phase_code=int(rec[6]) >> 32,
                        timestamp_us=int(rec[7]), duration_us=int(rec[11]),
                        num_devices=int(rec[12]), event_id=int(rec[13]))
                    encoded = json.dumps(row)
                    stream.write(encoded + '\n')
                    if client is not None:
                        published.append(encoded)
                    samples += 1
            stream.flush()
            if client is not None:
                with client.pipeline() as pipe:
                    key = f'online_rows_node{args.node_rank}'
                    if published:
                        pipe.rpush(key, *published)
                        pipe.ltrim(key, -10000, -1)
                    if len(dropped) > loss_start:
                        pipe.rpush(f'online_dropped_node{args.node_rank}',
                                   *(json.dumps(x) for x in dropped[loss_start:]))
                    # Publication is ordered before this ACK in one transaction.
                    pipe.set(f'online_drained_node{args.node_rank}',
                             json.dumps(dict(through_iteration=through_iteration)))
                    pipe.execute()
            if stopping or (output / 'drain.stop').exists():
                break
            time.sleep(.2)
    (output / 'drain_status.json').write_text(json.dumps(dict(samples=samples,
        dropped=dropped, cursors=cursor, snapshot='producer_mutex_fresh_records',
        status='complete' if not dropped else 'data_loss'), indent=2))
    if client is not None:
        client.close()
    probe_state.close(); state_file.close()


if __name__ == '__main__':
    main()
