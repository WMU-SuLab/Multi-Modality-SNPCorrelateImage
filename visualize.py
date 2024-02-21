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
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from init import init_net
from utils import setup_seed
from utils.gene import handle_gene_file
from utils.interpretability import make_img_cam, imshow, make_gene_saliency_maps, make_gene_image_saliency_maps, \
    img_saliency_maps_show
from utils.transforms import gene_image_transforms


@click.command()
@click.argument('model_name', type=str)
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--snp_numbers', type=int, default=0)
@click.option('--gene_file_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--gene_id_file_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--image_file_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--saliency_maps_name', type=str, default='IntegratedGradients')
@click.option('--baseline_method', type=str, default='gaussian_blur')
@click.option('--cam_name', type=str, default='GradCAM')
def main(model_name: str, wts_path: str,
         snp_numbers: int, gene_file_path: str, gene_id_file_path, image_file_path: str,
         saliency_maps_name: str, baseline_method: str, cam_name: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False
    setup_seed(2023)
    # 初始化网络
    net = init_net(device, model_name, snp_numbers, pretrain_checkpoint_path=wts_path)
    if image_file_path:
        image = Image.open(image_file_path)
        img_tensor = gene_image_transforms['test'](image)
        img_tensor = img_tensor.to(device)
    if gene_file_path and gene_id_file_path:
        df = pd.DataFrame()
        with open(gene_id_file_path) as f:
            df['gene_id'] = f.read().strip().split(',')[1:]
        gene_tensor = handle_gene_file(gene_file_path)
        df['gene_val'] = gene_tensor.tolist()
        gene_tensor = gene_tensor.to(device)
    if image_file_path and not gene_file_path:
        # img是HWC，需要变成 BCHW 的数据，B==1
        target_layers = [net.image_features.stages[-1][-1].dwconv]
        targets = [BinaryClassifierOutputTarget(1)]
        # targets = None
        img, grayscale_cam, visualization = make_img_cam(
            net, cam_name, target_layers, targets, img_tensor=img_tensor, use_cuda=use_cuda)
        imshow([img, grayscale_cam, visualization], ["image", "cam", "image + cam"],
               file_name=f'{cam_name}_{os.path.basename(image_file_path)}')
    elif not image_file_path and gene_file_path and gene_id_file_path:
        saliency_maps = make_gene_saliency_maps(net, saliency_maps_name, gene_tensor)
        df['weight'] = saliency_maps[0].flatten().tolist()
        df = df.sort_values(by=['weight_abs'], ascending=False)
        df.to_csv(os.path.join(os.path.dirname(gene_file_path),
                               f'{os.path.splitext(os.path.basename(gene_file_path))[0]}.csv'))
    elif image_file_path and gene_file_path:
        # attributions, delta = make_gene_image_saliency_maps(net, saliency_maps_name, img_tensor=img_tensor,
        #                                                     gene_tensor=gene_tensor)
        attributions = make_gene_image_saliency_maps(net, device, img_tensor=img_tensor, gene_tensor=gene_tensor,
                                                     saliency_maps_name=saliency_maps_name,
                                                     baseline_method=baseline_method)
        gene_attribution, image_attribution = attributions
        df['weight'] = gene_attribution.flatten().tolist()
        df['weight_abs'] = df['weight'].abs()
        df = df.sort_values(by='weight_abs', ascending=False)
        df.to_csv(os.path.join(os.path.dirname(gene_file_path),
                               f'{saliency_maps_name}_{os.path.splitext(os.path.basename(gene_file_path))[0]}.csv'))
        img_saliency_maps_show(image_attribution, img_tensor, image_file_path, saliency_maps_name)


if __name__ == '__main__':
    main()
