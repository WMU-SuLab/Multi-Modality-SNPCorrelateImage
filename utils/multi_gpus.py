# -*- encoding: utf-8 -*-
"""
@File Name      :   multi_gpus.py
@Create Time    :   2023/4/20 19:31
@Description    :   
@Version        :   
@License        :   MIT
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :   
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
"""
__auth__ = 'diklios'

import torch
import torch.distributed as dist


def init_distributed_mode(dist_backend, world_size, rank):
    # 初始化分布式环境，主要用来帮助进程间通信
    dist.init_process_group(backend=dist_backend, init_method='env://',
                            world_size=world_size, rank=rank)
    # 屏障，所有GPU都运行到这里的时候才会继续往下运行
    # 但是实际上 init_process_group() 里面已经有了 barrier()，所以这里可以不用
    dist.barrier()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_available()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def barrier():
    dist.barrier()


def synchronize(device):
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)


def cleanup():
    dist.destroy_process_group()
