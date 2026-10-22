"""Match probe cliques to explicitly reported process-group membership."""
from collections import defaultdict
import json


def build_topology(rank_reports, world_size):
    """Reports come from each rank's initialized process groups, before training."""
    groups = {}
    for report in rank_reports:
        for group in report['groups']:
            key = (group['name'], tuple(sorted(group['members'])),
                   group.get('label'), group.get('localization'))
            entry = groups.setdefault(key, dict(group, reporters=[]))
            entry['reporters'].append(report['rank'])
    return dict(schema_version=2, world_size=world_size,
                groups=list(groups.values()), rank_reports=rank_reports)


def load_topology(path):
    try:
        topology = json.loads(path.read_text())
        if not isinstance(topology.get('groups'), list):
            return None, [dict(reason='topology has no reported groups')]
        return topology, []
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return None, [dict(reason='topology unavailable', detail=str(error))]


def label_cliques(rows, topology):
    """Fail closed on incomplete observations, invalid groups, or semantic aliases."""
    diagnostics, valid = [], []
    if topology is None:
        return [], [dict(reason='no topology; probe rows excluded', samples=len(rows))]
    world_size = topology.get('world_size')
    groups, incomplete_members = [], []
    for group in topology.get('groups', []):
        if not isinstance(group, dict):
            diagnostics.append(dict(reason='invalid topology group', group=group))
            continue
        members = group.get('members', [])
        good = (isinstance(world_size, int) and world_size > 0 and
                isinstance(members, list) and len(members) > 1 and
                all(type(r) is int and 0 <= r < world_size for r in members) and
                len(set(members)) == len(members) and bool(group.get('name')) and
                type(group.get('localization')) is bool)
        reporters = group.get('reporters', [])
        complete = (isinstance(reporters, list) and all(type(r) is int for r in reporters)
                    and isinstance(members, list) and all(type(r) is int for r in members)
                    and set(reporters) == set(members))
        if not good or not complete:
            diagnostics.append(dict(reason='invalid or incomplete topology group', group=group))
            if isinstance(members, list) and all(type(r) is int for r in members):
                incomplete_members.append(members)
            continue
        groups.append(group)
    cliques = defaultdict(list)
    for row in rows:
        cliques[row['clique']].append(row)
    for clique, observations in cliques.items():
        members = sorted({row['rank'] for row in observations})
        sizes = {row['num_devices'] for row in observations}
        detail = dict(clique=clique, observed_members=members, communicator_sizes=sorted(sizes))
        if len(sizes) != 1 or sizes != {len(members)}:
            diagnostics.append(dict(reason='incomplete or inconsistent clique', **detail))
            continue
        if any(set(m) == set(members) for m in incomplete_members):
            diagnostics.append(dict(reason='incomplete topology match', **detail))
            continue
        matches = [g for g in groups if sorted(g['members']) == members]
        if len(matches) != 1:
            diagnostics.append(dict(reason='ambiguous topology match' if matches else 'no topology match',
                                    matching_groups=[g['name'] for g in matches], **detail))
            continue
        group = matches[0]
        if not group['localization'] or not group.get('label'):
            diagnostics.append(dict(reason='group excluded from localization', group=group['name'], **detail))
            continue
        for row in observations:
            row = dict(row)
            # Message sizes define comparable observations, not new groups.
            row.update(tag=group['label'], group_name=group['name'], group_members=members)
            valid.append(row)
    return valid, diagnostics
