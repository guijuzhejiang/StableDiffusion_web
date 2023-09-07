# coding=utf-8
# @Time : 2023/8/21 上午11:04
# @File : worker_proc.py
import multiprocessing as mp
import os

mp.set_start_method('spawn', force=True)
import sys
from lib.celery_workshop.wokrshop import WorkShop
import torch
torch.set_num_threads(1)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(sys.argv)
    is_cuda = bool(sys.argv[-1])
    del sys.argv[-1]
    op_name = str(sys.argv[-1])
    del sys.argv[-1]
    index = int(sys.argv[-1])
    del sys.argv[-1]
    print(sys.argv)
    WorkShop.instance_worker_proc(index, op_name, is_cuda)
    # ws = WorkShop(OperatorSD)
    # ws.instance_worker_proc(index, op_name, is_cuda)
