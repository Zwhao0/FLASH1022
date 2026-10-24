"""CPU-only online monitoring and bounded, iteration-aligned diagnosis."""
import argparse
import json
from pathlib import Path
import signal
import statistics
import time

from flash_v1.analysis import WindowDetector


def emit(output, event, **values):
    row = dict(event=event, wall_time=time.time(), **values)
    with (Path(output) / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def decode(value):
    return json.loads(value) if value is not None else None


class OnlineSchedule:
    """Rank zero schedules a request for the next eight complete iterations."""
    def __init__(self, client, output, profile_iterations=8, channel=None):
        if profile_iterations != 8:
            raise ValueError('frozen localization requires exactly eight profile iterations')
        self.client, self.output = client, Path(output)
        self.channel = channel
        if channel is None:
            self.output.mkdir(parents=True, exist_ok=True)
        self.schedule = None
        self.last_epoch = 0

    def next_epoch(self, iteration):
        request = (dict(epoch=self.channel.get_request()) if self.channel is not None
                   else decode(self.client.get('online_request')))
        if request and int(request['epoch']) > self.last_epoch:
            if self.schedule and iteration < self.schedule['end']:
                raise RuntimeError('overlapping online profile requests')
            self.last_epoch = int(request['epoch'])
            self.schedule = dict(epoch=self.last_epoch, start=iteration + 1, end=iteration + 8)
            if self.channel is not None:
                self.channel.set_schedule(**self.schedule)
            else:
                self.client.set(f'online_schedule_{self.last_epoch}', json.dumps(self.schedule))
                emit(self.output, 'profile_schedule', **self.schedule)
        if self.schedule and self.schedule['start'] <= iteration + 1 <= self.schedule['end']:
            return self.schedule['epoch']
        return 0


def window_rows(rows, schedule):
    return [row for row in rows if schedule['start'] <= int(row['iteration']) <= schedule['end']]


def missing_coverage(rows, schedule, world_size=16):
    observed = {(int(row['iteration']), int(row['rank'])) for row in rows}
    return [(step, rank) for step in range(schedule['start'], schedule['end'] + 1)
            for rank in range(world_size) if (step, rank) not in observed]


def drains_after(client, end_iteration):
    polls = [decode(client.get(f'online_drained_node{node}')) for node in range(2)]
    return all(poll is not None and int(poll['through_iteration']) >= end_iteration for poll in polls)


def diagnose(client, output, schedule, model):
    """Return None for evidence still arriving; never extend the sampling window."""
    from flash_v1.topology import label_cliques
    from flash_v1.wait_localize import CoverageError, localize_run
    rows = window_rows([decode(row) for node in range(2)
                        for row in client.lrange(f'online_rows_node{node}', 0, -1)], schedule)
    dropped = [decode(row) for node in range(2)
               for row in client.lrange(f'online_dropped_node{node}', 0, -1)]
    if dropped:
        return dict(predicted='inconclusive', reason='ring_data_loss', dropped=dropped)
    if missing_coverage(rows, schedule):
        return None
    output = Path(output)
    topology = json.loads((output / 'topology.json').read_text())
    rows, diagnostics = label_cliques(rows, topology)
    configuration = json.loads((output / 'configuration.json').read_text())
    configuration.update(attention_backward_block_size=3, sampling_stride=8)
    destination = output / f"online_profile_{schedule['epoch']}"
    destination.mkdir(exist_ok=True)
    manifest = dict(run_id=destination.name, configuration=configuration,
                    interval=dict(start=schedule['start'], end=schedule['end']),
                    status='completed', samples=len(rows))
    for name, value in [('manifest', manifest), ('topology', topology), ('profile', rows),
                        ('label_diagnostics', diagnostics)]:
        (destination / f'{name}.json').write_text(json.dumps(value))
    try:
        return localize_run(destination, model)
    except CoverageError:
        return None  # Retry only the drain, with a fixed ten-second deadline.


def add_monitor_sample(detector, iteration, duration, confirm_recent=False):
    """Optional persistence check after unchanged BEAST; rejected CP is uncommitted."""
    previous_cp = getattr(detector, 'last_cp', None)
    event = detector.add(iteration, duration)
    if event is not None and confirm_recent:
        recent = list(detector.samples)[-3:]
        contiguous = len(recent) == 3 and [r[0] for r in recent] == list(range(iteration-2, iteration+1))
        if (not contiguous or statistics.median(r[1] for r in recent)
                <= event['baseline_mean'] * (1 + detector.degradation)):
            detector.last_cp = previous_cp
            return None
    return event


def profile_transition_event(configuration, pending, result, epoch, max_profiles, latest, done):
    """One child per original request, emitted at the real completion boundary."""
    # V2 fusion can change the public rank; continuation remains the original
    # core decision. Auxiliary model output never populates this identity.
    core = result.get('profile_transition_core', result)
    if not isinstance(core, dict):return None
    hint = core.get('profile_transition')
    if (not configuration.get('monitor_profile_transition', False)
            or pending.get('parent_epoch') is not None or not hint
            or core.get('status') != 'localized' or hint.get('rank') != core.get('rank')
            or done is not None or epoch >= max_profiles
            or pending.get('schedule') is None or latest < pending['schedule']['end']):
        return None
    # The existing command path schedules two iterations after observation.
    steps = configuration.get('training_steps')
    if steps is None or latest + 9 > steps:
        return None
    return dict(epoch=epoch + 1, iteration=latest, source='profile_transition',
                parent_epoch=epoch, profile_iterations=8, transition=hint)


def bounded_diagnostic_timeout(client, output, schedule, model):
    """Optional versioned callback; the existing deadline and v1 stay unchanged."""
    resolver = vars(diagnose).get('timeout_result')
    if callable(resolver):
        result = resolver(client, output, schedule, model)
        if result is not None:return result
    return dict(predicted='inconclusive', reason='bounded_drain_or_coverage_timeout')


def controller(client, output, model, warmup=12, max_profiles=3, stopped=lambda: False):
    from flash_v1.monitor_channel import MonitorChannel
    path = Path(output) / 'monitor.bin'
    channel = MonitorChannel(path) if path.exists() else None
    try:
        _controller(client, output, model, warmup, max_profiles, stopped, channel)
    finally:
        if channel is not None:
            channel.close()


def _controller(client, output, model, warmup, max_profiles, stopped, channel):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detector = WindowDetector(warmup=warmup)
    configuration_path = output / 'configuration.json'
    configuration = json.loads(configuration_path.read_text()) if configuration_path.exists() else {}
    benchmark_blocks = configuration.get('monitor_benchmark_blocks')
    resume = configuration.get('monitor_resume') == 'contiguous'
    confirm_recent = configuration.get('monitor_confirm_recent_three', False)
    replay_rows = []
    last_iteration, epoch, pending, exhausted = 0, 0, None, False
    channel_cursor, forwarded_epoch = 0, 0
    client.set('online_pending', '0')
    while not stopped() and not client.exists('shutdown'):
        if channel is not None:
            observations, channel_cursor = channel.read_since(channel_cursor)
            if observations:
                with client.pipeline() as pipe:
                    pipe.rpush('online_iterations', *(json.dumps(row) for row in observations))
                    pipe.ltrim('online_iterations', -500, -1)
                    pipe.set('online_latest_iteration', observations[-1]['iteration'])
                    pipe.execute()
            schedule = channel.get_schedule()
            if schedule is not None and schedule['epoch'] > forwarded_epoch:
                forwarded_epoch = schedule['epoch']
                client.set(f'online_schedule_{forwarded_epoch}', json.dumps(schedule))
                emit(output, 'profile_schedule', **schedule)
        else:
            observations = [decode(encoded) for encoded in client.lrange('online_iterations', 0, -1)]
        observations = replay_rows + observations
        replay_rows = []
        for row in observations:
            iteration = int(row['iteration'])
            if iteration <= last_iteration:
                continue
            last_iteration = iteration
            if pending:
                if resume:
                    pending['resume_rows'].append(row)
                if pending.get('schedule') is None:
                    pending['schedule'] = decode(client.get(f'online_schedule_{epoch}'))
                schedule = pending['schedule']
                if schedule and iteration >= schedule['end'] and 'deadline' not in pending:
                    pending['deadline'] = time.monotonic() + 10
                continue
            if epoch >= max_profiles:
                if not exhausted:
                    emit(output, 'quota_exhausted', max_profiles=max_profiles)
                    exhausted = True
                continue
            if int(row.get('profile_epoch', 0)):
                if resume:
                    # BEAST assumes uniform unit spacing. A PROFILE gap must
                    # break its sample window, but must not erase last_cp.
                    detector.samples.clear()
                continue
            if benchmark_blocks is not None and not any(
                    block['mode'] == 'B' and block['start'] <= iteration <= block['end']
                    for block in benchmark_blocks):
                continue  # Controlled A blocks, never used by ordinary training.
            if resume and getattr(detector, 'last_iteration', None) is not None:
                if iteration != detector.last_iteration + 1:
                    detector.samples.clear()  # Lost/non-monitor intervals are gaps too.
            event = add_monitor_sample(detector, iteration, float(row['wall_iteration_ms']), confirm_recent)
            if event is not None:
                epoch += 1
                event = dict(event)
                event.pop('profile_seconds', None)
                event['profile_iterations'] = 8
                emit(output, 'detection', epoch=epoch, **event)
                client.set('online_pending', '1')
                client.set('online_request', json.dumps(dict(epoch=epoch)))
                if channel is not None:
                    channel.set_request(epoch)
                pending = dict(requested_at=time.monotonic(), schedule=None,
                               resume_after=iteration, resume_rows=[])
        if pending:
            if pending.get('schedule') is None:
                pending['schedule'] = decode(client.get(f'online_schedule_{epoch}'))
            schedule = pending['schedule']
            done = decode(client.get('online_training_done'))
            result = None
            if done is not None and schedule is None:
                result = dict(predicted='inconclusive', reason='training_ended_before_profile_start')
            elif done is not None and schedule['end'] > int(done['last_iteration']):
                result = dict(predicted='inconclusive', reason='truncated_profile')
            elif schedule is None:
                if time.monotonic() - pending['requested_at'] > 10:
                    raise TimeoutError('training did not acknowledge online profile request')
            elif max(last_iteration, int(done['last_iteration']) if done else 0) >= schedule['end']:
                # The schedule or completion marker may arrive after the final
                # iteration row; do not depend on another training row to start draining.
                pending.setdefault('deadline', time.monotonic() + 10)
                if drains_after(client, pending['schedule']['end']):
                    result = diagnose(client, output, pending['schedule'], model)
                if result is None and time.monotonic() >= pending['deadline']:
                    result = bounded_diagnostic_timeout(client, output, pending['schedule'], model)
            if result is not None:
                emit(output, 'localization', epoch=epoch, profile_start=schedule['start'] if schedule else None,
                     profile_end=schedule['end'] if schedule else None,
                     elapsed_seconds=time.monotonic() - pending['requested_at'], result=result)
                client.set(f'online_completed_{epoch}', json.dumps(result))
                latest_completed = last_iteration
                if configuration.get('monitor_profile_transition', False):
                    if channel is not None:
                        latest_completed = max(latest_completed, channel.latest_completed_iteration())
                    else:
                        latest = client.lrange('online_iterations', -1, -1)
                        if latest:
                            latest_completed = max(latest_completed, int(decode(latest[0])['iteration']))
                    done = decode(client.get('online_training_done'))
                transition = profile_transition_event(configuration, pending, result, epoch,
                                                      max_profiles, latest_completed, done)
                if resume:
                    # Revisit all rows observed while draining; PROFILE rows
                    # are barriers, ordinary rows remain eligible exactly once.
                    # Unread channel rows will arrive in the next poll normally.
                    replay_rows = pending['resume_rows']
                    last_iteration = pending['resume_after']
                else:
                    # Legacy behavior remains the default.
                    latest = client.lrange('online_iterations', -1, -1)
                    if latest:
                        last_iteration = max(last_iteration, int(decode(latest[0])['iteration']))
                    if done:
                        last_iteration = max(last_iteration, int(done['last_iteration']))
                    if transition is not None:
                        last_iteration = max(last_iteration, latest_completed)
                    detector = WindowDetector(warmup=last_iteration)
                pending = None
                if transition is not None:
                    epoch = transition['epoch']
                    emit(output, 'detection', **transition)
                    client.set('online_pending', '1')
                    client.set('online_request', json.dumps(dict(epoch=epoch)))
                    if channel is not None:
                        channel.set_request(epoch)
                    pending = dict(requested_at=time.monotonic(), schedule=None,
                                   resume_after=last_iteration, resume_rows=[],
                                   parent_epoch=transition['parent_epoch'])
                else:
                    client.set('online_pending', '0')
        client.set('online_processed_iteration', last_iteration)
        time.sleep(.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--redis-port', type=int, default=17422)
    parser.add_argument('--model', required=True)
    parser.add_argument('--warmup', type=int, default=12)
    parser.add_argument('--max-profiles', type=int, default=3)
    args = parser.parse_args()
    import redis
    from flash_v1.wait_localize import verify_model
    model = json.loads(Path(args.model).read_text())
    verify_model(model)
    stop = [False]
    def terminate(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    client = redis.Redis(port=args.redis_port, decode_responses=True)
    controller(client, args.output, model, args.warmup, args.max_profiles, lambda: stop[0])


if __name__ == '__main__':
    main()
