"""Physical layout for two eight-GPU nodes, each holding TP2 × PP2 × EP2."""
import os


_RANK_CACHES = (
    '_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS',
    '_DATA_PARALLEL_GLOBAL_RANKS',
    '_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP',
    '_CONTEXT_PARALLEL_GLOBAL_RANKS',
    '_PIPELINE_GLOBAL_RANKS',
    '_EMBEDDING_GLOBAL_RANKS',
    '_POSITION_EMBEDDING_GLOBAL_RANKS',
)


def physical_rank(logical):
    """Map vendor p*8 + replica*4 + e*2 + t to replica*8 + p*4 + e*2 + t.

    Swapping the PP and replica bits is its own inverse.
    """
    if not isinstance(logical, int) or not 0 <= logical < 16:
        raise ValueError('node-local layout requires a rank in [0, 16)')
    return (logical // 4 % 2) * 8 + (logical // 8) * 4 + logical % 4


class _Proxy:
    def __init__(self, wrapped, **overrides):
        self._wrapped = wrapped
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def initialize_node_local(state, torch, **init_kwargs):
    """Initialize vendor groups through a module-local rank mapping proxy.

    Requires torchrun's LOCAL_WORLD_SIZE=8 and world size 16. Call only for
    the explicit node-local mode; legacy initialization stays untouched.
    """
    dist = torch.distributed
    expected = dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2,
                    expert_model_parallel_size=2, context_parallel_size=1)
    if dist.get_world_size() != 16 or os.environ.get('LOCAL_WORLD_SIZE') != '8':
        raise ValueError('node-local layout requires world size 16 and LOCAL_WORLD_SIZE=8')
    for name, value in expected.items():
        if init_kwargs.get(name, 1) != value:
            raise ValueError(f'node-local layout requires {name}={value}')

    def get_rank(group=None):
        rank = dist.get_rank(group=group)
        world = getattr(getattr(dist, 'group', None), 'WORLD', None)
        return physical_rank(rank) if group is None or group is world else rank

    def new_group(ranks=None, *args, **kwargs):
        members = None if ranks is None else [physical_rank(rank) for rank in ranks]
        return dist.new_group(members, *args, **kwargs)

    original_torch = state.torch
    state.torch = _Proxy(torch, distributed=_Proxy(dist, get_rank=get_rank, new_group=new_group))
    try:
        result = state.initialize_model_parallel(**init_kwargs)
    finally:
        state.torch = original_torch
    for name in _RANK_CACHES:
        ranks = getattr(state, name, None)
        if ranks is not None:
            setattr(state, name, [physical_rank(rank) for rank in ranks])
    return result
