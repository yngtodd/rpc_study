from torch.distributed import rpc

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.remote(rref.owner(), _call_method, args=args, kwargs=kwargs)


def remote_sync(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def remote_async(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


def wait_for(futures):
    """ Wait for some futures returned by async call """
    return [f.wait() for f in list(futures)]


def async_wait(method, rrefs, args=None, kwargs=None):
    """ Run async call and wait for the futures """
    fut = [remote_async(method, rref, args=[rref]) for rref in rrefs]
    return wait_for(fut)


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    return [rpc.RRef(p) for p in module.parameters()]


def _call_method(method, rref, *args, **kwargs):
    r"""
    Helper function to call a remote method on an rref
    """
    return method(rref.local_value(), *args, **kwargs)
