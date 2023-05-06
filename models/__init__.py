# -*- encoding: utf-8 -*-
"""
@File Name      :   __init__.py.py
@Create Time    :   2023/2/22 22:05
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
from torch import nn

from .ConvNeXt.with_attention import convnext_tiny


class GeneNet(nn.Module):
    def __init__(self, snp_number: int):
        super(GeneNet, self).__init__()
        self.mlp = nn.Sequential(
            # 拼接 snp 和 image 特征
            nn.Linear(snp_number, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, snps):
        x = self.mlp(snps)
        return x


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        # 这个模型已经把最后一层分类去掉了，留下的是特征层
        self.image_features = convnext_tiny(1)
        self.mlp = nn.Sequential(
            # 拼接 snp 和 image 特征
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
        )  # 模型参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, image):
        x = self.image_features(image)
        y = self.mlp(x)
        return y

    def load_image_feature_state_dict(self, state_dict: dict):
        return self.image_features.load_state_dict(state_dict=state_dict['model'], strict=False)


class SNPImageNet(ImageNet):
    def __init__(self, snp_number: int):
        super(ImageNet, self).__init__()
        # 这个模型已经把最后一层分类去掉了，留下的是特征层
        self.image_features = convnext_tiny(1)

        self.mlp = nn.Sequential(
            # 拼接 snp 和 image 特征
            nn.Linear(768 + snp_number, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    def forward(self, snps, image):
        x = self.image_features(image)
        x = torch.cat([x, snps], dim=1)
        y = self.mlp(x)
        return y


nets = {
    f'{GeneNet.__name__}': GeneNet,
    f'{ImageNet.__name__}': ImageNet,
    f'{SNPImageNet.__name__}': SNPImageNet,
}


def nets_fake_data(device, model_name: str, batch_size: int, snp_numbers: int = 0):
    if model_name == GeneNet.__name__:
        return torch.randn(batch_size, snp_numbers).to(device)
    elif model_name == ImageNet.__name__:
        return torch.randn(batch_size, 3, 224, 224).to(device)
    elif model_name == SNPImageNet.__name__:
        return torch.randn(batch_size, snp_numbers).to(device), torch.randn(batch_size, 3, 224, 224).to(device)
    else:
        raise ValueError(f'No such model: {model_name}')


__all__ = ['nets', 'nets_fake_data']