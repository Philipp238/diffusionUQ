#!/bin/sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1   # if using InfiniBand and getting issues
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python main.py -c debug_pde.ini