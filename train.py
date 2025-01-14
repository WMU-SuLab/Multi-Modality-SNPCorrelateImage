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
from shutil import copytree, copy2

import click
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler

from base import checkpoints_dir, logs_dir, code_dir
from divide_dataset import mk_dataset_paths
from init import init_net, init_strategy
from models import nets_fake_data
from utils import setup_seed
from utils.dir import mk_dir
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from utils.records import experiments_record, train_epoch_record
from utils.time import datetime_now_str
from utils.workflow import workflows


@click.command()
@click.option('--seed', type=int, default=2023, help='随机种子')
@click.option('--train_dir_prefix', default=datetime_now_str(), type=str, help='train dir prefix')
@click.argument('model_name', type=str)
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.option('--snp_number', type=int, default=0, help='snp numbers')
@click.option('--pretrain_checkpoint_path', default=None, help='pretrain checkpoint path')
@click.option('--pretrain_image_feature_checkpoint_path', default=None, help='pretrain image feature checkpoint path')
@click.option('--dataset_in_memory', type=bool, is_flag=True, help='if store dataset in memory')
@click.option('--persistent_workers', type=bool, is_flag=True, default=True, help='if persistent workers')
@click.option('--gene_freq_file_path', type=click.Path(exists=True), default=None, help='gene freq file path')
@click.option('--label_data_id_field_name', type=str, default=None, help='label data id field name')
@click.option('--label_data_label_field_name', type=str, default=None, help='label data label field name')
@click.option('--epochs', default=10, type=int, help='epochs')
@click.option('--batch_size', default=8, type=int, help='batch size')
@click.option('--log_interval', default=1, type=int, help='save metrics interval')
@click.option('--save_interval', default=10, type=int, help='save wts interval')
@click.option('--lr', default=1e-4, type=float, help='learning rate')
@click.option('--step_size', default=10, type=int, help='step size')
@click.option('--gamma', default=0.1, type=float, help='gamma')
@click.option('--last_epoch', default=-1, type=int, help='last epoch')
@click.option('--use_early_stopping', default=True, help='if use early stopping')
@click.option('--early_stopping_step', default=7, type=int, help='early stopping step')
@click.option('--early_stopping_delta', default=0, type=int, help='early stopping delta')
@click.option('--cuda_device', default=0, type=int, help='cuda device')
@click.option('--remarks', default=None, help='remarks')
def main(
        seed: int, train_dir_prefix: str, model_name: str, dataset_dir_path: str, snp_number: int,
        pretrain_checkpoint_path: str, pretrain_image_feature_checkpoint_path: str,
        dataset_in_memory: bool, persistent_workers: bool,
        gene_freq_file_path: str, label_data_id_field_name: str, label_data_label_field_name: str,
        lr: float, step_size: int, gamma: float, last_epoch: int,
        epochs: int, batch_size: int, log_interval: int, save_interval: int,
        use_early_stopping: bool, early_stopping_step: int, early_stopping_delta: int,
        cuda_device: int, remarks: str
):
    print(f'当前时间:{train_dir_prefix}')
    if cuda_device:
        torch.cuda.set_device(cuda_device)
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
    print(f'current device:{device}')
    # 设置随机种子
    setup_seed(seed)
    # 读取数据集中snp的数量
    # with open(os.path.join(data_paths['train_gene_dir_path'],os.listdir(data_paths['train_gene_dir_path'])[0]), 'r') as f:
    #     text = f.read()
    #     snp_number=len(text.split(',')[1:])
    # 初始化网络
    net = init_net(device, model_name, snp_number, pretrain_checkpoint_path, pretrain_image_feature_checkpoint_path)
    # 初始化策略
    optimizer, scheduler, criterion, loss_early_stopping = init_strategy(
        net, lr, step_size, gamma, last_epoch, pretrain_checkpoint_path, early_stopping_step, early_stopping_delta)
    # 对实验做一些参数记录
    log_dir_path = os.path.join(logs_dir, train_dir_prefix)
    mk_dir(log_dir_path)
    experiments_record(
        os.path.join(log_dir_path, 'experiments_records.txt'),
        train_dir_prefix, model_name, dataset_dir_path, snp_number,
        epochs, batch_size, lr, step_size, criterion, gamma, save_interval, log_interval, remarks)
    # 创建权重和参数记录文件夹
    model_checkpoints_dir = os.path.join(checkpoints_dir, train_dir_prefix)
    mk_dir(model_checkpoints_dir)
    best_model_checkpoints_path = os.path.join(model_checkpoints_dir, 'best_model_checkpoints.pth')
    # 创建 tensorboard 日志文件夹
    writer = SummaryWriter(log_dir=log_dir_path)
    writer.add_graph(net, nets_fake_data(device, model_name, batch_size, snp_number))
    # 复制代码
    code_dir_path = os.path.join(code_dir, train_dir_prefix)
    mk_dir(code_dir_path)
    copytree('./models', os.path.join(code_dir_path, 'models'))
    copytree('./utils', os.path.join(code_dir_path, 'utils'))
    copy2('./train.py', os.path.join(code_dir_path, 'train.py'))
    copy2('./base.py', os.path.join(code_dir_path, 'base.py'))
    copy2('./init.py', os.path.join(code_dir_path, 'init.py'))
    copy2('./divide_dataset.py', os.path.join(code_dir_path, 'divide_dataset.py'))
    os.symlink('./work_dirs', os.path.join(code_dir_path, 'work_dirs'), target_is_directory=True)
    # 加载数据集
    # attention:需要使用dataset模块中的方法从原始数据中生成数据集，否则需要自己手动更改以下 dataloader 的各个文件和文件夹路径
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders_func = mk_data_loaders_single_funcs[model_name]
    data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': batch_size}
    if dataset_in_memory:
        data_loaders_func_kwargs['in_memory'] = dataset_in_memory
    if persistent_workers:
        data_loaders_func_kwargs['persistent_workers'] = persistent_workers
    if gene_freq_file_path:
        data_loaders_func_kwargs['gene_freq_file_path'] = gene_freq_file_path
    if label_data_id_field_name:
        data_loaders_func_kwargs['label_data_id_field_name'] = label_data_id_field_name
    if label_data_label_field_name:
        data_loaders_func_kwargs['label_data_label_field_name'] = label_data_label_field_name
    data_loaders = data_loaders_func(**data_loaders_func_kwargs)
    # 初始化参数
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_f1 = 0
    since = time.time()
    for epoch in range(epochs):
        # 训练一次、验证一次
        for phase in ['train', 'valid']:
            epoch_loss, all_metrics = workflows[phase](
                device, net, criterion, optimizer, tqdm.tqdm(data_loaders[phase]), data_loaders, phase,
                scaler=scaler
            )
            # 记录指标
            f1, best_f1, best_model_wts = train_epoch_record(
                epoch_loss, all_metrics, net, optimizer, epoch, epochs, phase,
                writer, log_interval, best_f1, best_model_wts, best_model_checkpoints_path, since)
            # 判断是否早停
            if use_early_stopping and phase == 'valid':
                loss_early_stopping(epoch_loss)
        # 一个epoch调整一次学习率，不能是一个batch，否则学习率会太小
        scheduler.step()
        if epoch % step_size == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'f1': f1,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoints_dir, f'epoch_{epoch}_model_checkpoints.pth'))
        if use_early_stopping and loss_early_stopping.early_stop:
            break
    # 训练全部完成
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f}'.format(best_f1))
    # 加载最佳模型权重，最后做一次总的测试
    net.load_state_dict(best_model_wts)
    if data_loaders.get('test', None):
        workflows['test'](device, net, data_loaders['test'], writer)
    else:
        workflows['test'](device, net, data_loaders['valid'], writer)


if __name__ == '__main__':
    # 查看GPU是否可用
    train_on_gpu = torch.cuda.is_available()
    print("GPU是否可用：", train_on_gpu)
    if train_on_gpu:
        print('CUDA is available!')
        # GPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        # os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
        # gpu_devices_number = os.environ["CUDA_VISIBLE_DEVICES"]
        # gpu_default_device_number = int(gpu_devices_number.split(',')[0])
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # tensor.to("cuda") = tensor.cuda()
        gpu_device = torch.device("cuda")
        # gpu_device = torch.device("cuda:0")
        # 查看GPU数量
        gpu_count = torch.cuda.device_count()
        print("GPU数量:", gpu_count)
        # torch方法查看CUDA版本
        print("CUDA版本:", torch.version.cuda)
        # 查看GPU索引号
        # print("GPU索引号：", torch.cuda.current_device())
        # 设置默认设备
        # torch.cuda.set_device(gpu_default_device_number)
        # 根据索引号得到GPU名称
        # print("GPU名称：", torch.cuda.get_device_name(gpu_default_device_number))
    else:
        print('CUDA is not available. Training on CPU ...')
        # CPU
        # tensor_gpu.to("cpu") = tensor_gpu.cpu()
        cpu_device = torch.device("cpu")
    main()
