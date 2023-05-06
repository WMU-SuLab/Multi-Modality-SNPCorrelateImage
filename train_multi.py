# -*- encoding: utf-8 -*-
"""
@File Name      :   train_distribution.py
@Create Time    :   2023/4/19 20:28
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

import copy
import os
import time

import click
import torch
import tqdm
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.tensorboard import SummaryWriter

from base import project_dir_path, checkpoints_dir, weights_dir, logs_dir
from divide_dataset import mk_dataset_paths
from init import init_net, init_strategy
from models import nets_fake_data
from utils import setup_seed
from utils.dir import mk_dir
from utils.finish import finish_train
from utils.metrics import count_metrics_binary_classification
from utils.mk_data_loaders import mk_train_multi_data_loaders_funcs
from utils.multi_gpus import init_distributed_mode, barrier, reduce_value,cleanup
from utils.records import experiments_record, train_epoch_record
from utils.time import datetime_now
from utils.workflow import workflows


@click.command()
@click.option('--rank', type=int, default=0, help='主机的编号')
@click.option('--local-rank', type=int, default=0, help='GPU的编号')
@click.option('--world-size', type=int, default=1, help='GPU的数量')
@click.option('--dist-backend', type=str, default='nccl', help='通信后端')
@click.option('--seed', type=int, default=2023, help='随机种子')
@click.argument('model_name', type=str)
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.option('--snp_numbers', type=int, default=0, help='snp numbers')
@click.option('--epochs', default=10, help='epochs')
@click.option('--batch_size', default=8, help='batch size')
@click.option('--lr', default=1e-4, help='learning rate')
@click.option('--step_size', default=10, help='step size')
@click.option('--gamma', default=0.1, help='gamma')
@click.option('--log_interval', default=1, help='save metrics interval')
@click.option('--save_interval', default=10, help='save wts interval')
@click.option('--pretrain_wts_path', default=None, help='pretrain weights path')
@click.option('--pretrain_image_feature_checkpoint_path', default=None, help='pretrain image feature checkpoint path')
@click.option('--use_early_stopping', default=True, help='if use early stopping')
@click.option('--early_stopping_step', default=5, help='early stopping step')
@click.option('--early_stopping_delta', default=0, help='early stopping delta')
@click.option('--train_dir_prefix', default=datetime_now(), type=str, help='train dir prefix')
@click.option('--remarks', default=None, help='remarks')
def main(
        rank, local_rank: int, world_size: int, dist_backend: str, seed: int,
        model_name: str, dataset_dir_path: str, snp_numbers: int,
        epochs: int, batch_size: int, lr: float, step_size: int, gamma: float,
        log_interval: int, save_interval: int,
        pretrain_wts_path: str, pretrain_image_feature_checkpoint_path: str,
        use_early_stopping: bool, early_stopping_step: int, early_stopping_delta: int,
        train_dir_prefix: str, remarks: str
):
    """
    :param rank: 单机情况下，主机的编号和GPU的编号是相同的
    :param local_rank:
    :param world_size: 使用了几块GPU，等于--nproc_per_node参数
    :param dist_backend:
    :param seed:
    :param model_name:
    :param dataset_dir_path:
    :param snp_numbers:
    :param epochs:
    :param batch_size:
    :param lr:
    :param step_size:
    :param gamma:
    :param log_interval:
    :param save_interval:
    :param pretrain_wts_path:
    :param pretrain_image_feature_checkpoint_path:
    :param use_early_stopping:
    :param early_stopping_step:
    :param early_stopping_delta:
    :param train_dir_prefix:
    :param remarks:
    :return:
    """
    # 初始化分布式训练
    world_size = int(os.environ.get('WORLD_SIZE', world_size))
    rank = int(os.environ.get('RANK', rank))
    local_rank = int(os.environ.get('LOCAL_RANK', local_rank))
    # 通信后端 nvidia GPU推荐使用 NCCL
    init_distributed_mode(dist_backend, world_size, rank)
    # 每个进程根据自己的local_rank设置应该使用的GPU
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    # 设置随机种子
    setup_seed(seed)
    # 初始化网络
    net = init_net(device, model_name, snp_numbers, pretrain_wts_path, pretrain_image_feature_checkpoint_path)
    net = ddp(net, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    # 初始化策略
    optimizer, scheduler, criterion, loss_early_stopping = init_strategy(
        net, lr, step_size, gamma, early_stopping_step, early_stopping_delta)
    # 对实验做一些参数记录
    if rank == 0:
        experiments_record(
            os.path.join(project_dir_path, 'experiments_records.txt'),
            train_dir_prefix, model_name, dataset_dir_path, snp_numbers,
            epochs, batch_size, lr, step_size, criterion, gamma, save_interval, log_interval, remarks)
    # 创建权重和参数记录文件夹
    model_checkpoint_dir = os.path.join(checkpoints_dir, train_dir_prefix)
    if rank == 0:
        mk_dir(model_checkpoint_dir)
    best_model_checkpoint_path = os.path.join(model_checkpoint_dir, 'best_model_checkpoint.pth')
    # 创建权重记录文件夹
    model_wts_dir = os.path.join(weights_dir, train_dir_prefix)
    if rank == 0:
        mk_dir(model_wts_dir)
    best_model_wts_path = os.path.join(model_wts_dir, 'best_model_wts.pth')
    # 创建 tensorboard 日志文件夹
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(logs_dir, train_dir_prefix), comment='multi-gpu-training')
        writer.add_graph(net, nets_fake_data(device, model_name, batch_size, snp_numbers))
    # 加载数据集
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders, samplers = mk_train_multi_data_loaders_funcs[model_name](data_paths, batch_size)
    # 初始化参数
    best_model_wts = copy.deepcopy(net.state_dict())
    best_f1 = 0
    since = time.time()
    for epoch in range(epochs):
        # 训练一次、验证一次
        for phase in ['train', 'valid']:
            samplers[phase].set_epoch(epoch)
            if phase == 'train':
                # 训练
                net.train()
            else:
                # 验证
                net.eval()
            # 循环所有数据
            data_loader_iter = data_loaders[phase]
            if rank == 0:
                data_loader_iter = tqdm.tqdm(data_loader_iter)
                data_loader_iter.set_description(f'Epoch {epoch + 1}/{epochs}, running on {phase} dataset')
            running_loss, y_true, y_pred, y_score = workflows[phase](
                device, net, criterion, optimizer, data_loader_iter, phase, multi_gpu=True)
            torch.cuda.synchronize(device)
            # 计算损失
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # 计算指标
            all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
            all_metrics = torch.tensor(all_metrics).to(device)
            barrier()
            all_metrics = reduce_value(all_metrics)
            all_metrics = all_metrics.cpu().numpy().tolist()
            # 记录指标
            if rank == 0:
                best_f1, best_model_wts = train_epoch_record(
                    epoch_loss, all_metrics, net, optimizer, epoch, epochs, phase,
                    writer, log_interval, best_f1, best_model_wts, best_model_checkpoint_path, since)
            # 判断是否早停
            if use_early_stopping and phase == 'valid':
                loss_early_stopping(epoch_loss)
        scheduler.step()
        if epoch % step_size == 0 and rank == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if epoch % save_interval == 0 and rank == 0:
            torch.save(net.state_dict(), os.path.join(model_wts_dir, f'epoch_{epoch}_model_wts.pth'))
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoint_dir, f'epoch_{epoch}_model_checkpoints.pth'))
        if use_early_stopping and loss_early_stopping.early_stop:
            break
    if rank == 0:
        finish_train(device, net, data_loaders, writer, best_f1, best_model_wts, best_model_wts_path, since)
        cleanup()


if __name__ == '__main__':
    """
    设置Shell变量：CUDA_VISIBLE_DEVICES='0,1...'
    设置使用哪些GPU：手动设置环境变量：os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6...'
    实际torch代码中的GPU号还是从0开始的，这个变量只是表示拿哪些GPU来训练，通过nvidia-smi可以看到具体的GPU号
    这种写法一般要放最代码的前面，但是这个写法非常的不pythonic，所以还是推荐使用环境变量的方式
    但是在Windows下设置环境变量不方便的时候，可以使用在代码中手动设置的方式
    
    单机多GPU训练：
    使用方法：python -m torch.distributed.launch --nproc_per_node=需要使用的GPU数量 train_multi.py
    使用方法2（推荐）：torchrun --nproc_per_node=需要使用的GPU数量 train_multi.py
    
    正常情况下，torch.distributed.launch 会自动添加环境变量和命令参数
    添加了--use_env 参数之后，在命令后面就不会添加命令参数了，只能通过环境变量来设置参数
    
    
    使用多GPU的时候，如果不是自己手动终止脚本，torch.distributed 给的端口不会默认释放，为了防止妨碍别人下次使用
    所以需要手动释放端口：ps -ef | grep 脚本名称 | grep -v grep | awk '{print "kill -9 "$2}' | sh
    
    多机训练的时候，需要设置 --nnodes 和 --master-port，非多机情况下，rank和local_rank是一样的
    """
    main()
