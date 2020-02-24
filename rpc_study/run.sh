#!/bin/sh

# Test PyTorch RPC on a single host using torch.distributed.launch

if [ $# -lt 1 ]; then
    echo "This should be launched using the python -m rpc_study.launcher <path>."
fi

python -m torch.distributed.launch \
    --nproc_per_node 3 \
    --use_env \
    $1
