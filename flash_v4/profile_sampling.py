"""Compute PROFILE sampling only; old plans retain stride16."""
def profile_stride(value=None):
    if value is None:
        return 16
    if value not in ('16', '32', '64'):
        raise ValueError('PROFILE sample stride must be16,32 or64')
    return int(value)
