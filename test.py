# -*- encoding: utf-8 -*-
"""
@File Name      :   test.py
@Create Time    :   2023/2/6 9:30
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

import click
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import setup_seed
from divide_dataset import mk_dataset_paths
from init import init_net
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from utils.workflow import workflows


@click.command()
@click.argument('model_name', type=str)
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('snp_number', type=int)
@click.argument('log_dir', type=click.Path(exists=True))
@click.option('--batch_size', default=8, help='batch size')
def main(model_name: str, dataset_dir_path, checkpoint_path, snp_number: int, log_dir, batch_size):
    setup_seed(2023)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = init_net(device, model_name, snp_number, pretrain_checkpoint_path=checkpoint_path)
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders = mk_data_loaders_single_funcs[model_name](data_paths, batch_size=batch_size)
    writer = SummaryWriter(log_dir=log_dir)
    net.eval()
    # 开始测试
    if data_loaders.get('test', None):
        workflows['test'](device, net, data_loaders['test'], writer)
    else:
        workflows['test'](device, net, data_loaders['valid'], writer)


if __name__ == '__main__':
    main()
