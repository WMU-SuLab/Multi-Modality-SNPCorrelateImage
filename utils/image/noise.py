# -*- encoding: utf-8 -*-
"""
@File Name      :   noise.py
@Create Time    :   2023/8/1 20:07
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@other information
"""
__auth__ = 'diklios'

import torch


def image_random_noise(img_tensor: torch.Tensor, prob: float = 0.05) -> torch.Tensor:
    c, h, w = img_tensor.shape
    mask = torch.rand([c, h, w]) < prob
    noise_tensor = torch.rand([c, h, w])
    img_tensor[mask] = noise_tensor[mask]
    return img_tensor


def gaussian_noise(img_tensor: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
    # c, h, w = img_tensor.shape
    # return torch.randn([c, h, w]) * std + mean
    return torch.randn_like(img_tensor) * std + mean


def image_gaussian_noise(img_tensor: torch.Tensor,mean: float = 0, std: float = 0.05) -> tuple[torch.Tensor, ...]:
    noise_tensor = gaussian_noise(img_tensor, mean, std)
    noise_img_tensor = img_tensor + noise_tensor
    for i in range(img_tensor.shape[0]):  # min-max normalization
        noise_tensor[i] = (noise_tensor[i] - noise_tensor[i].min()) / (noise_tensor[i].max() - noise_tensor[i].min())
        noise_img_tensor[i] = (noise_img_tensor[i] - noise_img_tensor[i].min()) / (
                noise_img_tensor[i].max() - noise_img_tensor[i].min())
    return noise_tensor, noise_img_tensor
