import threading

import torch
from torch import optim
from torch.distributed import rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

from rpc_study.rpc import run_study, Env
from rpc_study.rpc import *

"""
def remote_method(method, obj_rref, *args, **kwargs):
    return rpc.remote(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs,
    )
"""

def rpc_async_method(method, obj_rref, *args, **kwargs):
    """
    Call rpc.rpc_async on a method in a remote object.
    Args:
        method: the method (for example, Class.method)
        obj_rref (RRef): remote reference to the object
        args: positional arguments to pass to the method
        kwargs: keyword arguments to pass to the method
    Returns a Future to the method call result.
    """
    return rpc.rpc_async(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs,
    )


class MyModule:
    lock = threading.Lock()

    def __init__(self):
        # avoid race where 2 modules could be initialized
        # concurrently thus changing the order random numbers are drawn.
        with MyModule.lock:
            torch.manual_seed(0)
            self.w = torch.rand((3, 3), requires_grad=True)

    def forward(self, t1):
        return torch.mm(self.w, t1)

    def get_w(self):
        return self.w


def study():
    module1 = MyModule()
    module2 = MyModule()
    params = [module1.get_w(), module2.get_w()]
    local_optim = optim.SGD(params, lr=0.05)

    old_w1 = module1.w.clone().detach()
    old_w2 = module2.w.clone().detach()

    torch.manual_seed(0)
    t1 = torch.rand((3, 3), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    output1 = module1.forward(t2)
    output2 = module2.forward(output1)
    loss = torch.add(output2, t1).sum()

    loss.backward()
    local_optim.step()

    # distributed version
    owner1 = "worker%d" % ((Env.rank + 1) % Env.world_size)
    owner2 = "worker%d" % ((Env.rank + 2) % Env.world_size)

    remote_module1 = rpc.remote(owner1, MyModule)
    remote_module2 = rpc.remote(owner2, MyModule)
    remote_param1 = remote_method(MyModule.get_w, remote_module1)
    remote_param2 = remote_method(MyModule.get_w, remote_module2)

    old_w1_remote = remote_param1.to_here()

    dist_optim = DistributedOptimizer(
        optim.SGD, [remote_param1, remote_param2], lr=0.05
    )

    with dist_autograd.context():
        torch.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
        output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
        loss = torch.add(output2.wait(), t1)

        dist_autograd.backward([loss.sum()])
        dist_optim.step()

        new_w1 = rpc_async_method(MyModule.get_w, remote_module1).wait()
        new_w2 = rpc_async_method(MyModule.get_w, remote_module2).wait()


if __name__=='__main__':
    run_study(study)
