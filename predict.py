# -*- encoding: utf-8 -*-
"""
@File Name      :   predict.py
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

import os.path

import click
import torch
from PIL import Image

from init import init_net
from utils.gene import handle_gene_file
from utils.transforms import gene_image_transforms


@click.command()
@click.argument('model_name', type=str)
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--snp_numbers', type=int, default=0)
@click.option('--gene_file_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--image_file_path', type=click.Path(exists=True, dir_okay=False), default=None)
def main(model_name: str, wts_path: str, snp_numbers: int, gene_file_path: str, image_file_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化网络
    net = init_net(device, model_name, snp_numbers, pretrain_wts_path=wts_path)
    inputs = []
    if gene_file_path:
        if os.path.exists(gene_file_path) and os.path.isfile(gene_file_path):
            gene_data = handle_gene_file(gene_file_path)
            gene_data = gene_data.unsqueeze(0)
            gene_data = gene_data.to(device)
            inputs.append(gene_data)
        else:
            raise ValueError(f'invalid gene_file_path:{gene_file_path}')
    if image_file_path:
        if os.path.exists(image_file_path) and os.path.isfile(image_file_path):
            image = Image.open(image_file_path)
            image = gene_image_transforms['test'](image)
            image = image.unsqueeze(0)
            image = image.to(device)
            inputs.append(image)
        else:
            raise ValueError(f'invalid image_file_path:{image_file_path}')
    if not inputs:
        raise ValueError('no data')
    # 开始预测
    net.eval()
    with torch.no_grad():
        output = net(*inputs)
        y_pred = torch.sigmoid(output).gt(0.5).int().reshape(-1).tolist()
        print(y_pred)


if __name__ == '__main__':
    main()
