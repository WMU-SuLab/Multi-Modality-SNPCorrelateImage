# -*- encoding: utf-8 -*-
"""
@File Name      :   visualize_dataset_custom.py   
@Create Time    :   2024/1/5 22:46
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@Other Info     :
"""
__auth__ = 'diklios'

import os

import click
import numpy as np
import pandas as pd
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib import pyplot as plt
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from divide_dataset import mk_dataset_paths
from init import init_net
from utils import setup_seed
from utils.image import tensor2numpy
from utils.mk_data_loaders import mk_data_loaders_single_funcs


@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('snp_numbers', type=int)
@click.argument('dataset_dir_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('snp_col_name_file_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
def main(checkpoint_path: str, snp_numbers: int, dataset_dir_path: str, snp_col_name_file_path: str,
         output_dir_path: str):
    if not output_dir_path:
        output_dir_path = os.path.dirname(checkpoint_path)
    image_weights_dir_path = os.path.join(output_dir_path, 'image_weights')
    if not os.path.exists(image_weights_dir_path):
        os.makedirs(image_weights_dir_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(2023)
    # 初始化网络
    net = init_net(device, 'SNPImageNet', snp_numbers, pretrain_checkpoint_path=checkpoint_path)
    net.eval()
    ig = IntegratedGradients(net)
    print(f'dataset_dir_path:{dataset_dir_path}')
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders_func = mk_data_loaders_single_funcs['SNPImageNet']
    data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': 32, 'use_eye_side': True}
    data_loaders = data_loaders_func(**data_loaders_func_kwargs)
    df = pd.DataFrame()
    with open(snp_col_name_file_path) as f:
        df['gene_id'] = f.read().strip().split(',')[1:]
    df['weight_abs'] = 0
    df['weight'] = 0
    df['weight_plus'] = 0
    df['weight_minus'] = 0
    image_OS = np.zeros((224, 224))
    image_OS_attr = np.zeros((3, 224, 224))
    image_OS_count = 0
    image_OD = np.zeros((224, 224))
    image_OD_attr = np.zeros((3, 224, 224))
    image_OD_count = 0
    count = 0
    for phase in ['train', 'valid']:
        if phase == 'train':
            for inputs, labels, eye_sides in tqdm(data_loaders[phase]):
                snp_tensors, img_tensors = inputs
                for i in range(len(snp_tensors)):
                    count += 1
                    snp_tensor = snp_tensors[i]
                    img_tensor = img_tensors[i]
                    eye_side = int(eye_sides[i])
                    snp_baseline1 = []
                    snp_baseline2 = []
                    for snp in snp_tensor:
                        if int(snp) == 0:
                            snp_baseline1.append(2)
                            snp_baseline2.append(2)
                        elif int(snp) == 2:
                            snp_baseline1.append(0)
                            snp_baseline2.append(0)
                        elif int(snp) == -1:
                            snp_baseline1.append(-1)
                            snp_baseline2.append(-1)
                        if int(snp) == 1:
                            snp_baseline1.append(0)
                            snp_baseline2.append(2)
                    baselines1 = tuple([torch.tensor(snp_baseline1, dtype=torch.float).unsqueeze(0).to(device),
                                        gaussian_blur(img_tensor, kernel_size=[7, 7], sigma=[0.1, 2.0]).unsqueeze(
                                            0).to(device)])
                    baselines2 = tuple([torch.tensor(snp_baseline2, dtype=torch.float).unsqueeze(0).to(device),
                                        gaussian_blur(img_tensor, kernel_size=[7, 7], sigma=[0.1, 2.0]).unsqueeze(
                                            0).to(device)])
                    inputs_unsqueezed = tuple([snp_tensor.unsqueeze(0).to(device), img_tensor.unsqueeze(0).to(device)])

                    attributions1, delta1 = ig.attribute(inputs_unsqueezed, baselines1, n_steps=10,
                                                         return_convergence_delta=True)
                    gene_attribution1, image_attribution1 = attributions1
                    attributions2, delta2 = ig.attribute(inputs_unsqueezed, baselines2, n_steps=10,
                                                         return_convergence_delta=True)
                    gene_attribution2, image_attribution2 = attributions2
                    # 计算 SNP 权重
                    snp_weight1 = gene_attribution1.flatten().cpu().numpy()
                    snp_weight2 = gene_attribution2.flatten().cpu().numpy()
                    df['weight'] += (snp_weight1 + snp_weight2) / 2
                    weight_abs = (np.abs(snp_weight1) + np.abs(snp_weight2)) / 2
                    df['weight_abs'] += weight_abs
                    new_snp_weight1_plus = np.zeros_like(snp_weight1)
                    new_snp_weight1_minus = np.zeros_like(snp_weight1)
                    new_snp_weight1_plus[snp_weight1 > 0] = snp_weight1[snp_weight1 > 0]
                    new_snp_weight1_minus[snp_weight1 < 0] = snp_weight1[snp_weight1 < 0]
                    new_snp_weight2_plus = np.zeros_like(snp_weight2)
                    new_snp_weight2_minus = np.zeros_like(snp_weight2)
                    new_snp_weight2_plus[snp_weight2 > 0] = snp_weight2[snp_weight2 > 0]
                    new_snp_weight2_minus[snp_weight2 < 0] = snp_weight2[snp_weight2 < 0]
                    df['weight_plus'] += new_snp_weight1_plus + new_snp_weight2_plus
                    df['weight_minus'] += new_snp_weight1_minus + new_snp_weight2_minus
                    # 计算 OS/OD 权重
                    image_attribution1 = image_attribution1.squeeze().detach().cpu().numpy()
                    if eye_side == 1:
                        image_OS_attr += image_attribution1
                    else:
                        image_OD_attr += image_attribution1
                    plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(image_attribution1, (1, 2, 0)),
                                                                          tensor2numpy(img_tensor),
                                                                          methods=["original_image", "heat_map"],
                                                                          signs=["all", "absolute_value"],
                                                                          show_colorbar=True,
                                                                          outlier_perc=1,
                                                                          use_pyplot=False
                                                                          )
                    plt_fig.savefig(
                        os.path.join(image_weights_dir_path, f'image_{count}_{eye_side}_attribution1_viz.png'))
                    image_attribution1 = np.sum(np.abs(image_attribution1), axis=0)
                    image_attribution1 = (image_attribution1 - image_attribution1.min()) / \
                                         (image_attribution1.max() - image_attribution1.min())
                    image_attribution1 = (image_attribution1 * 255).astype('uint8')
                    image_attribution2 = image_attribution2.squeeze().detach().cpu().numpy()
                    if eye_side == 1:
                        image_OS_attr += image_attribution2
                    else:
                        image_OD_attr += image_attribution2
                    plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(image_attribution2, (1, 2, 0)),
                                                                          tensor2numpy(img_tensor),
                                                                          methods=["original_image", "heat_map"],
                                                                          signs=["all", "absolute_value"],
                                                                          show_colorbar=True,
                                                                          outlier_perc=1,
                    use_pyplot=False
                                                                          )
                    plt_fig.savefig(
                        os.path.join(image_weights_dir_path, f'image_{count}_{eye_side}_attribution2_viz.png'))
                    image_attribution2 = np.sum(np.abs(image_attribution2), axis=0)
                    image_attribution2 = (image_attribution2 - image_attribution2.min()) / \
                                         (image_attribution2.max() - image_attribution2.min())
                    image_attribution2 = (image_attribution2 * 255).astype('uint8')

                    image_attribution = (image_attribution1 + image_attribution2) / 2
                    if eye_side == 1:
                        image_OS_count += 1
                        image_OS += image_attribution
                    else:
                        image_OD_count += 1
                        image_OD += image_attribution
    image_OS = (image_OS / image_OS_count).astype('uint8')
    image_OD = (image_OD / image_OD_count).astype('uint8')
    image_OS_attr = image_OS_attr / image_OS_count / 2
    image_OD_attr = image_OD_attr / image_OD_count / 2
    image_OS_img = Image.fromarray(image_OS)
    image_OS_img.save(os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}_OS.png'))
    image_OD_img = Image.fromarray(image_OD)
    image_OD_img.save(os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}_OD.png'))
    image_OS_img_heatmap = plt.get_cmap('hot')(image_OS)
    plt.imsave(
        os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}_OS_heatmap.png'),
        image_OS_img_heatmap)
    image_OD_img_heatmap = plt.get_cmap('hot')(image_OD)
    plt.imsave(
        os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}_OD_heatmap.png'),
        image_OD_img_heatmap)
    plt_fig_OS, plt_axis_OS = viz.visualize_image_attr(np.transpose(image_OS_attr, (1, 2, 0)),use_pyplot=False)
    plt_fig_OS.savefig(
        os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}_OS_attr.png'))
    plt_fig_OD, plt_axis_OD = viz.visualize_image_attr(np.transpose(image_OD_attr, (1, 2, 0)),use_pyplot=False)
    plt_fig_OD.savefig(
        os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}_OD_attr.png'))

    df = df.sort_values(by='weight_abs', ascending=False)
    df.to_csv(os.path.join(output_dir_path, f'IntegratedGradients_{os.path.basename(dataset_dir_path)}.csv'),
              index=False)


if __name__ == '__main__':
    main()
