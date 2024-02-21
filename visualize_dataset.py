# -*- encoding: utf-8 -*-
"""
@File Name      :   visualize_dataset.py   
@Create Time    :   2023/11/14 9:50
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

from divide_dataset import mk_dataset_paths
from init import init_net
from utils import setup_seed
from utils.interpretability import make_gene_image_saliency_maps
from utils.mk_data_loaders import mk_data_loaders_single_funcs


@click.command()
@click.argument('model_name', type=str)
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('snp_numbers', type=int)
@click.argument('dataset_dir_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('gene_id_file_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('--saliency_maps_name', type=str, default='IntegratedGradients')
@click.option('--baseline_method', type=str, default='gaussian_blur')
def main(model_name: str, wts_path: str,
         snp_numbers: int, dataset_dir_path: str, gene_id_file_path: str,
         output_dir_path: str, saliency_maps_name: str, baseline_method: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(2023)
    # 初始化网络
    net = init_net(device, model_name, snp_numbers, pretrain_wts_path=wts_path)
    net.eval()
    print(f'dataset_dir_path:{dataset_dir_path}')
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders_func = mk_data_loaders_single_funcs[model_name]
    data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': 32}
    data_loaders = data_loaders_func(**data_loaders_func_kwargs)
    df = pd.DataFrame()
    with open(gene_id_file_path) as f:
        df['gene_id'] = f.read().strip().split(',')[1:]
    df['weight_abs'] = 0
    df['weight'] = 0
    for phase in ['train', 'valid']:
        if phase == 'train':
            for inputs, labels in data_loaders[phase]:
                inputs = [each_input.to(device) for each_input in inputs]
                gene_tensors, img_tensors = inputs
                for i in range(len(gene_tensors)):
                    gene_tensor = gene_tensors[i]
                    img_tensor = img_tensors[i]
                    attributions = make_gene_image_saliency_maps(net, device, img_tensor=img_tensor, gene_tensor=gene_tensor,
                                                                 saliency_maps_name=saliency_maps_name,
                                                                 baseline_method=baseline_method)
                    gene_attribution, image_attribution = attributions
                    weight = gene_attribution.flatten().cpu().numpy()
                    df['weight'] += weight
                    weight_abs = np.abs(weight)
                    df['weight_abs'] += weight_abs
    # todo：按照标签记录值
    df = df.sort_values(by='weight_abs', ascending=False)
    print(dataset_dir_path)
    df.to_csv(os.path.join(output_dir_path, f'{saliency_maps_name}_{os.path.basename(dataset_dir_path)}.csv'))


if __name__ == '__main__':
    r"""
    example: python .\visualize_dataset.py SNPImageNet .\work_dirs\records\weights\20231106193010\best_model_wts.pth 
    11058 .\work_dirs\data\divide\20231106125540\ 
    .\work_dirs\data\gene\students_snps_all_frequency_0.01\selected_genes_percentile_10\columns.csv .\work_dirs\test\
    """
    main()
