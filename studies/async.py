import threading

import torch
from torch import optim
from torch.distributed import rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

from rpc_study.rpc import run_study, Env
from rpc_study.rpc import *


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

        output1 = remote_async(MyModule.forward, remote_module1, t2)
        output2 = remote_async(MyModule.forward, remote_module2, output1.wait())
        loss = torch.add(output2.wait(), t1)

        dist_autograd.backward([loss.sum()])
        dist_optim.step()

        new_w1 = remote_async(MyModule.get_w, remote_module1).wait()
        new_w2 = remote_async(MyModule.get_w, remote_module2).wait()

        print(f'Old weight vs new weight: {old_w1 == new_w1}')
        print(f'Old weight vs new weight: {old_w2 == new_w2}')

        w1_consistent = (new_w1 == module1.get_w()).all()
        w2_consistent = (new_w2 == module2.get_w()).all()

        print(f'w1 consist: {w1_consistent}')
        print(f'w2 consist: {w2_consistent}')


if __name__=='__main__':
    run_study(study)
