"""Explicit CPU-only auxiliary MONITOR crossover selector; absent in normal plans."""
def validate_blocks(blocks):
    if not isinstance(blocks, list):
        raise ValueError('auxiliary benchmark blocks must be a list')
    seen=set()
    for block in blocks:
        if (not isinstance(block,dict) or not isinstance(block.get('id'),str)
                or not block['id'] or type(block.get('start')) is not int
                or type(block.get('end')) is not int or block['start']<1
                or block['end']<block['start'] or type(block.get('aux_enabled')) is not bool):
            raise ValueError('invalid auxiliary benchmark block')
        steps=set(range(block['start'],block['end']+1))
        if seen & steps:
            raise ValueError('overlapping auxiliary benchmark blocks')
        seen.update(steps)
    return blocks


def active_auxiliary(auxiliary, blocks, iteration, epoch):
    for block in blocks:
        if block['start']<=iteration<=block['end']:
            if epoch:
                raise ValueError('PROFILE entered declared MONITOR benchmark block')
            return auxiliary if block['aux_enabled'] else None
    return auxiliary
