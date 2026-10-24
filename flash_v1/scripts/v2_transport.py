"""Supervisor-side incremental evidence transport; never reads fault plans."""
import json


class EvidencePublisher:
    def __init__(self, output, node_rank, client, publish_completion=False, *,
                 filename='evidence_rank{rank}.jsonl', key_prefix='online_evidence_node',
                 completed_record='iteration_complete', completion_prefix='online_drained_node'):
        self.output, self.node_rank, self.client = output, node_rank, client
        self.offsets, self.partial = {}, {}
        self.published = 0
        self.publish_completion = publish_completion
        self.completed = {}
        self.filename, self.key_prefix = filename, key_prefix
        self.completed_record, self.completion_prefix = completed_record, completion_prefix

    def publish(self):
        lines = []
        offsets, partial = dict(self.offsets), dict(self.partial)
        for rank in range(self.node_rank * 8, (self.node_rank + 1) * 8):
            path = self.output / self.filename.format(rank=rank)
            if not path.exists():
                continue
            with path.open() as stream:
                stream.seek(self.offsets.get(rank, 0))
                chunk = stream.read()
                offsets[rank] = stream.tell()
            data = self.partial.get(rank, '') + chunk
            pieces = data.split('\n')
            partial[rank] = pieces.pop()
            for line in pieces:
                if line:
                    row = json.loads(line)
                    if row.get('rank') != rank:
                        raise ValueError('evidence rank/file mismatch')
                    lines.append(line)
        if lines:
            self.client.rpush(f'{self.key_prefix}{self.node_rank}', *lines)
            self.published += len(lines)
            for line in lines:
                row = json.loads(line)
                if row.get('record_type') == self.completed_record:
                    rank = int(row['rank'])
                    self.completed[rank] = max(self.completed.get(rank, 0), int(row['iteration']))
            if self.publish_completion and len(self.completed) == 8:
                # The unchanged controller gates diagnosis on both node ACKs.
                # Exact epoch/window/row completeness is checked by online_v2.
                self.client.set(f'{self.completion_prefix}{self.node_rank}', json.dumps(dict(
                    through_iteration=min(self.completed.values()), backend='semantic_only')))
        # A publish failure leaves all local cursors uncommitted and aborts the
        # owned supervisor. Do not retry an ambiguous Redis append automatically.
        self.offsets, self.partial = offsets, partial
        return len(lines)
