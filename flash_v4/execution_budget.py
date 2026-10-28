"""Explicit execution-only opt-in for the two R61 long healthy sessions."""

LONG_HEALTH = 'r61_long_health_v1'


def long_health_budget(row):
    name = row.get('execution_budget')
    if name is None:
        return None
    if (name != LONG_HEALTH or type(row.get('steps')) is not int
            or row['steps'] != 240 or type(row.get('timeout')) is not int
            or row['timeout'] != 360 or row.get('mode') != 'online'
            or row.get('force_profile_at', 0) != 0
            or row.get('fault_actor') is not False):
        raise ValueError('R61 execution budget requires 240 healthy online steps and timeout360')
    return dict(name=LONG_HEALTH, steps=240, node_timeout_seconds=360,
                controller_seconds=440, telemetry_groups=80,
                telemetry_bytes=8 * 1024 * 1024, sample_interval_seconds=6,
                monitor_worker_seconds=360)


def monitor_worker_seconds(name):
    if name is None:
        return 150
    if name != LONG_HEALTH:
        raise ValueError('unknown online execution budget')
    return 360


def validate_node_budget(name, steps, timeout, mode, actor=False, forced=False):
    if name is None:
        if not 1 <= steps <= 120 or not 1 <= timeout <= 600:
            raise ValueError('invalid bounded compatibility run')
        return None
    if actor or forced:
        raise ValueError('long healthy execution excludes actors and forced PROFILE')
    return long_health_budget(dict(execution_budget=name, steps=steps,
                                   timeout=timeout, mode=mode, fault_actor=False))
