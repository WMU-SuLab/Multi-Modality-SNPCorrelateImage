# -*- encoding: utf-8 -*-
"""
@File Name      :   train.py
@Create Time    :   2022/10/24 16:45
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
from torch.utils.tensorboard import SummaryWriter

from base import project_dir_path, checkpoints_dir, weights_dir, logs_dir
from divide_dataset import mk_dataset_paths
from init import init_net, init_strategy
from models import nets_fake_data
from utils import setup_seed
from utils.dir import mk_dir
from utils.finish import finish_train
from utils.metrics import count_metrics_binary_classification
from utils.mk_data_loaders import mk_data_loaders_funcs
from utils.records import experiments_record, train_epoch_record
from utils.time import datetime_now
from utils.workflow import workflows


@click.command()
@click.option('--seed', type=int, default=2023, help='随机种子')
@click.argument('model_name', type=str)
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.option('--snp_numbers', type=int, default=0, help='snp numbers')
@click.option('--gene_freq_file_path', type=click.Path(exists=True), default=None, help='gene freq file path')
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
@click.option('--early_stopping_step', default=7, help='early stopping step')
@click.option('--early_stopping_delta', default=0, help='early stopping delta')
@click.option('--train_dir_prefix', default=datetime_now(), type=str, help='train dir prefix')
@click.option('--remarks', default=None, help='remarks')
def main(
        seed: int, model_name: str, dataset_dir_path: str, snp_numbers: int, gene_freq_file_path: str,
        epochs: int, batch_size: int, lr: float, step_size: int, gamma: float,
        log_interval: int, save_interval: int,
        pretrain_wts_path: str, pretrain_image_feature_checkpoint_path: str,
        use_early_stopping: bool, early_stopping_step: int, early_stopping_delta: int,
        train_dir_prefix: str, remarks: str
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置随机种子
    setup_seed(seed)
    # 初始化网络
    net = init_net(device, model_name, snp_numbers, pretrain_wts_path, pretrain_image_feature_checkpoint_path)
    # 初始化策略
    optimizer, scheduler, criterion, loss_early_stopping = init_strategy(
        net, lr, step_size, gamma, early_stopping_step, early_stopping_delta)
    # 对实验做一些参数记录
    experiments_record(
        os.path.join(project_dir_path, 'experiments_records.txt'),
        train_dir_prefix, model_name, dataset_dir_path, snp_numbers,
        epochs, batch_size, lr, step_size, criterion, gamma, save_interval, log_interval, remarks)
    # 创建权重和参数记录文件夹
    model_checkpoint_dir = os.path.join(checkpoints_dir, train_dir_prefix)
    mk_dir(model_checkpoint_dir)
    best_model_checkpoint_path = os.path.join(model_checkpoint_dir, 'best_model_checkpoint.pth')
    # 创建权重记录文件夹
    model_wts_dir = os.path.join(weights_dir, train_dir_prefix)
    mk_dir(model_wts_dir)
    best_model_wts_path = os.path.join(model_wts_dir, 'best_model_wts.pth')
    # 创建 tensorboard 日志文件夹
    writer = SummaryWriter(log_dir=os.path.join(logs_dir, train_dir_prefix))
    writer.add_graph(net, nets_fake_data(device, model_name, batch_size, snp_numbers))
    # 加载数据集
    # attention:需要使用dataset模块中的方法从原始数据中生成数据集，否则需要自己手动更改以下 dataloader 的各个文件和文件夹路径
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders_func = mk_data_loaders_funcs[model_name]
    data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': batch_size}
    if gene_freq_file_path:
        data_loaders_func_kwargs['gene_freq_file_path'] = gene_freq_file_path
    data_loaders = data_loaders_func(**data_loaders_func_kwargs)
    # 初始化参数
    best_model_wts = copy.deepcopy(net.state_dict())
    best_f1 = 0
    since = time.time()
    for epoch in range(epochs):
        # 训练一次、验证一次
        for phase in ['train', 'valid']:
            if phase == 'train':
                # 训练
                net.train()
            else:
                # 验证
                net.eval()
            # 循环所有数据
            data_loader_iter = tqdm.tqdm(data_loaders[phase])
            running_loss, y_true, y_pred, y_score = workflows[phase](
                device, net, criterion, optimizer, data_loader_iter, phase)
            # 计算损失
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # 计算指标
            all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
            # 记录指标
            best_f1, best_model_wts = train_epoch_record(
                epoch_loss, all_metrics, net, optimizer, epoch, epochs, phase,
                writer, log_interval, best_f1, best_model_wts, best_model_checkpoint_path, since)
            # 判断是否早停
            if use_early_stopping and phase == 'valid':
                loss_early_stopping(epoch_loss)
        scheduler.step()
        if epoch % step_size == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if epoch % save_interval == 0:
            torch.save(net.state_dict(), os.path.join(model_wts_dir, f'epoch_{epoch}_model_wts.pth'))
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoint_dir, f'epoch_{epoch}_model_checkpoints.pth'))
        if use_early_stopping and loss_early_stopping.early_stop:
            break
    finish_train(device, net, data_loaders, writer, best_f1, best_model_wts, best_model_wts_path, since)


if __name__ == '__main__':
    # CPU
    # tensor_gpu.to("cpu") = tensor_gpu.cpu()
    cpu_device = torch.device("cpu")
    # GPU
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # tensor.to("cuda:0") = tensor.cuda()
    gpu_device = torch.device("cuda:0")
    # 查看GPU是否可用
    train_on_gpu = torch.cuda.is_available()
    # print("GPU是否可用：", train_on_gpu)
    # # 查看GPU数量
    # gpu_count = torch.cuda.device_count()
    # print("GPU数量：", gpu_count)
    # # torch方法查看CUDA版本
    # print("torch方法查看CUDA版本：", torch.version.cuda)
    # # 查看GPU索引号
    # print("GPU索引号：", torch.cuda.current_device())
    # # 根据索引号得到GPU名称
    # print("GPU名称：", torch.cuda.get_device_name(0))
    if train_on_gpu:
        print('CUDA is available!')
    else:
        print('CUDA is not available. Training on CPU ...')
    main()
