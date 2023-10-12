# -*- encoding: utf-8 -*-
"""
@File Name      :   interpretability.py
@Create Time    :   2023/4/10 18:30
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

import matplotlib.pyplot as plt
import numpy as np
import pytorch_grad_cam as grad_cams
import torch
import torchcam.methods as torch_cams
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, visualization as viz
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, gaussian_blur
from matplotlib import cm
from utils.image.noise import image_random_noise, image_gaussian_noise
from .image import tensor2numpy


def imshow(images, titles=None, file_name="test.jpg", dir_path: str = './', size=6):
    lens = len(images)
    fig = plt.figure(figsize=(size * lens, size))
    if not titles:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(images[i - 1].shape) == 2:
            plt.imshow(images[i - 1], cmap='Reds')
        else:
            plt.imshow(images[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(dir_path, f'{file_name}'), bbox_inches='tight')


class ReshapeTransform:
    def __init__(self, height: int = 14, width: int = 14):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        result = tensor.reshape(tensor.size(0), self.height, self.width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result


def make_img_cam(net, cam_name, target_layers: list, targets: list = None, img_tensor=None,
                 use_cuda=False):
    """
    :param net:
    :param cam_name:
    :param target_layers: 如果传入多个layer，cam输出结果将会取均值
    :param targets:
    :param img_tensor: 需要是经过transform的图
    :param use_cuda:
    :return:
    """
    if not img_tensor:
        raise ValueError('please input img tensor or gene tensor')
    input_tensor = img_tensor.unsqueeze(0)

    # 如果传入多个layer，cam输出结果将会取均值
    # reshape_transform 是当输出的feature map不是不是 BCHW 的形式时，需要进行reshape
    if CAM := getattr(grad_cams, cam_name, None):
        cam = CAM(model=net, target_layers=target_layers, use_cuda=use_cuda)
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        # targets=None 自动调用概率最大的类别显示
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        # grayscale_cams 根据 targets 的数量，会返回多个热力图
        # for grayscale_cam in grayscale_cams:
        grayscale_cam = grayscale_cams[0, :]
        img = tensor2numpy(img_tensor)
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        return img, grayscale_cam, visualization
    elif CAM := getattr(torch_cams, cam_name, None):
        with CAM(net, target_layers) as cam_extractor:
            # Preprocess your data and feed it to the model
            out = net(input_tensor)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        img = tensor2numpy(img_tensor)
        visualization = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'),
                                     alpha=0.5)
        return img, activation_map[0].squeeze(0).numpy(), visualization
    else:
        raise ValueError(f'{cam_name} does not exist.')


def img_saliency_maps_show(image_attribution, img_tensor, image_file_path, saliency_maps_name):
    image_attribution_norm = np.transpose(image_attribution.detach().cpu().squeeze().numpy(), (1, 2, 0))
    # 设置配色方案
    # default_cmap = LinearSegmentedColormap.from_list('custom',
    #                                                  [(0, '#ffffff'),
    #                                                   (0.25, '#000000'),
    #                                                   (1, '#000000')], N=256)
    cmap = cm.get_cmap('jet')
    # 可视化 IG 值
    dir_path = os.path.dirname(image_file_path)
    file_name = os.path.basename(image_file_path)
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,  # 224,224,3
                                                          tensor2numpy(img_tensor),  # 224,224,3
                                                          # np.transpose(img_tensor.squeeze().cpu().detach().numpy(),
                                                          #              (1, 2, 0)),
                                                          methods=["original_image", "heat_map", "heat_map", "heat_map",
                                                                   "heat_map", ],
                                                          signs=["all", "all", "absolute_value", "positive",
                                                                 "negative", ],
                                                          fig_size=(16, 6),
                                                          # cmap=default_cmap,
                                                          cmap=cmap,
                                                          show_colorbar=True,
                                                          outlier_perc=1)
    plt_fig.savefig(os.path.join(dir_path, f'multiple_heat_map_{saliency_maps_name}_{file_name}'), bbox_inches='tight')
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,  # 224,224,3
                                                          tensor2numpy(img_tensor),  # 224,224,3
                                                          # np.transpose(img_tensor.squeeze().cpu().detach().numpy(),
                                                          #              (1, 2, 0)),
                                                          methods=["original_image", "blended_heat_map",
                                                                   "blended_heat_map",
                                                                   "blended_heat_map", "blended_heat_map"],
                                                          signs=["all", "all", "absolute_value", "positive",
                                                                 "negative", ],
                                                          fig_size=(16, 6),
                                                          # cmap=default_cmap,
                                                          cmap=cmap,
                                                          show_colorbar=True,
                                                          outlier_perc=1)
    plt_fig.savefig(os.path.join(dir_path, f'multiple_blended_heat_map_{saliency_maps_name}_{file_name}'),
                    bbox_inches='tight')
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,  # 224,224,3
                                                          tensor2numpy(img_tensor),  # 224,224,3
                                                          # np.transpose(img_tensor.squeeze().cpu().detach().numpy(),
                                                          #              (1, 2, 0)),
                                                          methods=["original_image", "masked_image",
                                                                   "masked_image", "masked_image"],
                                                          signs=["all", "absolute_value", "positive", "negative", ],
                                                          fig_size=(16, 6),
                                                          # cmap=default_cmap,
                                                          cmap=cmap,
                                                          show_colorbar=True,
                                                          outlier_perc=1)
    plt_fig.savefig(os.path.join(dir_path, f'multiple_masked_image_{saliency_maps_name}_{file_name}'),
                    bbox_inches='tight')
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,  # 224,224,3
                                                          tensor2numpy(img_tensor),  # 224,224,3
                                                          # np.transpose(img_tensor.squeeze().cpu().detach().numpy(),
                                                          #              (1, 2, 0)),
                                                          methods=["original_image", "alpha_scaling",
                                                                   "alpha_scaling", "alpha_scaling"],
                                                          signs=["all", "absolute_value", "positive", "negative", ],
                                                          fig_size=(16, 6),
                                                          # cmap=default_cmap,
                                                          cmap=cmap,
                                                          show_colorbar=True,
                                                          outlier_perc=1)
    plt_fig.savefig(os.path.join(dir_path, f'multiple_alpha_scaling_{saliency_maps_name}_{file_name}'),
                    bbox_inches='tight')



def make_saliency_maps(net, device, inputs: tuple[torch.Tensor, ...], saliency_maps_name: str = 'IntegratedGradients',
                       baseline_method: str = 'gaussian_blur'):
    inputs_unsqueezed = tuple([item.unsqueeze(0) for item in inputs])

    net.eval()
    if saliency_maps_name == 'Saliency':
        sa = Saliency(net)
        attributions = sa.attribute(inputs, target=0, abs=False)
    elif saliency_maps_name == 'IntegratedGradients':
        # 零噪声/黑图
        if baseline_method == 'zeros':
            baselines = tuple([torch.zeros_like(item).to(device) for item in inputs_unsqueezed])
        # 灰度图
        elif baseline_method == 'gray':
            baselines = tuple([torch.ones_like(item).to(device) * 0.5 for item in inputs_unsqueezed])
        # 白色
        elif baseline_method == 'white':
            baselines = tuple([torch.ones_like(item).to(device) for item in inputs_unsqueezed])
        # 随机噪声
        elif baseline_method == 'random':
            baselines = tuple([torch.rand_like(item).to(device) for item in inputs_unsqueezed])
        elif baseline_method == 'random_image':
            baselines = tuple([torch.zeros_like(inputs_unsqueezed[0]).to(device),
                               image_random_noise(inputs[1], prob=0.05).unsqueeze(0).to(device)])  # 高斯噪声
        elif baseline_method == 'gaussian':
            baselines = tuple([torch.randn_like(item).to(device) for item in inputs_unsqueezed])
        elif baseline_method == 'gaussian_image':
            baselines = tuple([torch.zeros_like(inputs_unsqueezed[0]).to(device),
                               image_gaussian_noise(inputs[1])[1].unsqueeze(0).to(device)])
        # 高斯模糊
        elif baseline_method == 'gaussian_blur':
            baselines = tuple([torch.zeros_like(inputs_unsqueezed[0]).to(device),
                               gaussian_blur(inputs[1], kernel_size=[7, 7], sigma=[0.1, 2.0]).unsqueeze(0).to(device)])
        else:
            raise ValueError('no this baseline method')

        ig = IntegratedGradients(net)
        attributions, delta = ig.attribute(inputs_unsqueezed, baselines, target=0, n_steps=100,
                                           return_convergence_delta=True)
        print('IG Attributions:', attributions)
        print('Convergence Delta:', delta)
        # return attributions, delta
        # return attributions, delta, baselines
    elif saliency_maps_name == 'NoiseTunnel':
        ig = IntegratedGradients(net)
        noise_tunnel = NoiseTunnel(ig)
        attributions = noise_tunnel.attribute(inputs, target=0, nt_samples=12, nt_type='smoothgrad_sq', )
    else:
        raise ValueError('no this saliency maps method')
    return attributions


def make_gene_saliency_maps(net, device, gene_tensor: torch.Tensor, saliency_maps_name, baseline_method):
    return make_saliency_maps(net, device, (gene_tensor,), saliency_maps_name, baseline_method)


def make_gene_image_saliency_maps(net, device, gene_tensor: torch.Tensor, img_tensor: torch.Tensor, saliency_maps_name,
                                  baseline_method, ):
    return make_saliency_maps(net, device, (gene_tensor, img_tensor), saliency_maps_name, baseline_method)
