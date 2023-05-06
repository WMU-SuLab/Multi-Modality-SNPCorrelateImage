# -*- encoding: utf-8 -*-
"""
@File Name      :   visualize.py
@Create Time    :   2023/4/27 10:30
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

import os

import click
import torch
from PIL import Image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from init import init_net
from utils.cam import make_cam, imshow
from utils.transforms import gene_image_transforms


@click.command()
@click.argument('model_name', type=str)
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--snp_numbers', type=int, default=0)
@click.option('--gene_file_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--image_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--cam_name', type=str, default='GradCAM')
def main(model_name: str, wts_path: str, snp_numbers: int, gene_file_path: str, image_path: str, cam_name: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False
    # 初始化网络
    net = init_net(device, model_name, snp_numbers, pretrain_wts_path=wts_path)
    if gene_file_path and image_path:
        return
    if image_path:
        # img是HWC，需要变成 BCHW 的数据，B==1
        image = Image.open(image_path)
        img_tensor = gene_image_transforms['test'](image)
        img_tensor = img_tensor.to(device)
        target_layers = [net.image_features.stages[-1][-1].dwconv]
        targets = [BinaryClassifierOutputTarget(1)]
        # targets = None
        img, grayscale_cam, visualization = make_cam(cam_name, net, target_layers, targets, img_tensor, use_cuda)
        imshow([img, grayscale_cam, visualization], ["image", "cam", "image + cam"],
               file_name=f'{cam_name}_{os.path.basename(image_path)}')
    if gene_file_path:
        return


if __name__ == '__main__':
    main()
