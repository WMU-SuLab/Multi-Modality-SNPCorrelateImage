# -*- encoding: utf-8 -*-
"""
@File Name      :   try.py
@Create Time    :   2022/11/28 10:51
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
from divide_dataset import mk_dataset_paths
from init import init_net
from utils import setup_seed
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from captum.attr import IntegratedGradients
from torchvision.transforms.functional import gaussian_blur
from PIL import Image
import numpy as np
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from utils.image import tensor2numpy
if __name__ == '__main__':
    checkpoint_path = r"F:\sunhj\Multi-Modality-SNPCorrelateImage\work_dirs\records\checkpoints\20240210201520\best_model_checkpoints.pth"
    snp_numbers = 15043
    dataset_dir_path = r"F:\sunhj\Multi-Modality-SNPCorrelateImage\work_dirs\data\divide\20240201120611"

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

    for inputs, labels, eye_side in data_loaders['train']:
        snp_tensors, img_tensors = inputs
        for i in range(len(snp_tensors)):
            snp_tensor = snp_tensors[i]
            img_tensor = img_tensors[i]
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
            weight1 = gene_attribution1.flatten().cpu().numpy()
            # print(image_attribution1)
            # print(image_attribution1.shape)
            image_attribution1 = image_attribution1.squeeze().detach().cpu().numpy()
            plt_fig, plt_axis = viz.visualize_image_attr(
                np.transpose(image_attribution1, (1, 2, 0)),
                # tensor2numpy(img_tensor),
                method='heat_map', )
            plt_fig.savefig('image_attribution1_viz.png')
            image_attribution1 = np.sum(np.abs(image_attribution1),axis=0)
            print(image_attribution1.shape)
            # image_attribution1 = image_attribution1.squeeze().detach().cpu().numpy()
            # print(image_attribution1)
            # print(image_attribution1.shape)
            image_attribution1 = (image_attribution1 - image_attribution1.min()) / \
                                 (image_attribution1.max() - image_attribution1.min())
            image_attribution1 = (image_attribution1 * 255).astype('uint8')
            print(image_attribution1.shape)
            image_attribution1_img = Image.fromarray(image_attribution1)
            image_attribution1_img.save('image_attribution1.png')
            image_attribution1_heatmap = plt.get_cmap('hot')(image_attribution1)
            plt.imsave('image_attribution1_heatmap.png', image_attribution1_heatmap)

            # print(image_attribution1_heatmap.shape)
            # import cv2
            # cv2.imwrite('image_attribution1_heatmap.png', np.transpose(image_attribution1_heatmap, (1, 2, 0)) *  255)
            break
        break
