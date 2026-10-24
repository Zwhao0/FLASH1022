"""Static instrumentation identity, derived before training and without fault state."""
def observed_context(observer):
    if observer is None:
        return 'plain'
    name=type(observer).__name__
    if name=='ScopedComputeSlowdown':
        if not getattr(observer,'handles',None):
            raise ValueError('scoped observer has no installed scope hooks')
        return 'scoped_dispatch'
    if name=='WholeComputeSlowdown':
        return 'whole_dispatch'
    raise ValueError('unknown training instrumentation observer')
