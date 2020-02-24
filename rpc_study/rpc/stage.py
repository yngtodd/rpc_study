import os

import torch
from torch.distributed import rpc


def run_study(study):
    """
    Setup RPC and run a study.

    Args:
        study: (callable) function that runs 
        some minimal example.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))

    if rank == 0: 
        print(f'Initializing study on rank {rank}')
        rpc.init_rpc("study", rank=rank, world_size=world_size)
        study()
    else:
        print(f'Initializing worker on rank {rank}')
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # block until all rpcs finish
    rpc.shutdown()
