"""Post-run process telemetry interpretation; never a FLASH detector input."""
import csv
import io


def vllm_activity(apps, pmon):
    names = {}
    if apps.get('returncode') == 0:
        for row in csv.reader(io.StringIO(apps.get('stdout', '')), skipinitialspace=True):
            if len(row) >= 4 and row[1].strip().isdigit():
                names[int(row[1])] = row[2].strip()
    active, observed = [], []
    available = pmon.get('returncode') == 0
    if available:
        for line in pmon.get('stdout', '').splitlines():
            row = line.split()
            if len(row) < 5 or row[0].startswith('#') or not row[1].isdigit():
                continue
            pid = int(row[1])
            name = names.get(pid, '') + ' ' + row[-1]
            if 'vllm' not in name.lower() or not row[0].isdigit() or not row[3].isdigit():
                continue
            item = dict(gpu=int(row[0]), pid=pid, process_name=name.strip(), sm_percent=int(row[3]))
            observed.append(item)
            if item['sm_percent'] > 0:
                active.append(item)
    return dict(pmon_available=available, measured_worker_rows=len(observed),
                active_worker_rows=active, activity_observed=bool(active),
                interpretation='sampled concurrent activity only; absence is not proof of no interference')
