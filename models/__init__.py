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
from torchvision.models import resnet18, vit_b_16, convnext_tiny
from .Image import convnext_tiny as convnext_tiny_self

from .Image import ConvNeXtTiny, RETFoundNet
from .MultiModal import ConcatMultiModalNet, SelfAttentionMultiModalNet, TransformerMultiModalNet, \
    MultiHeadAttentionMultiModalNet, CrossAttentionMultiModalNet
from .SNP import SimpleLinear, DoubleLayerSimpleMLP, ResMLP, CapsuleNet, DeepExGRS


class SNPNet(nn.Module):
    def __init__(self, snp_number: int):
        super(SNPNet, self).__init__()
        self.snp_number = snp_number
        self.snp_model = SimpleLinear(snp_number)
        # self.snp_model = DoubleLayerSimpleMLP(snp_number)
        # self.snp_model = ResMLP(snp_number)
        # CapsuleNet需要改dataset输出数据、loss函数、workflow
        # self.snp_model = CapsuleNet(snp_number)
        # self.snp_model = DeepExGRS(snp_number)

    def forward(self, snps):
        x = self.snp_model(snps)
        return x


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        # self.image_model = resnet18(num_classes=1)
        # self.image_model = resnet18(weights=True)
        # self.image_model.fc = nn.Linear(512, 1)
        # self.image_model = ConvNeXtTiny()
        self.image_model = convnext_tiny(weights=True)
        self.image_model.classifier[-1] = nn.Linear(768, 1)
        # self.image_model = vit_b_16(weights=True)
        # self.image_model.heads = nn.Linear(768, 1)
        # self.image_model = RETFoundNet()

    def forward(self, image):
        y = self.image_model(image)
        return y

    def load_image_feature_state_dict(self, state_dict: dict):
        return self.image_model.load_image_model_state_dict(state_dict['model'])
        # return self.image_model.load_image_model_state_dict(state_dict)


class SNPImageNet(nn.Module):
    def __init__(self, snp_number: int):
        super(SNPImageNet, self).__init__()
        self.multi_modal_model = ConcatMultiModalNet(snp_number)
        # self.multi_modal_model = SelfAttentionMultiModalNet(snp_number)
        # self.multi_modal_model = MultiHeadAttentionMultiModalNet(snp_number)
        # self.multi_modal_model = TransformerMultiModalNet(snp_number)
        # self.multi_modal_model = CrossAttentionMultiModalNet(snp_number)

    def forward(self, snps, image):
        y = self.multi_modal_model(snps, image)
        return y


# class SNPImageNet(nn.Module):
#     def __init__(self, snp_number: int):
#         super(SNPImageNet, self).__init__()
#         self.snp_number = snp_number
#
#         self.image_features = nn.Sequential(
#             convnext_tiny_self(1),
#             nn.BatchNorm1d(768),
#             nn.ReLU(),
#         )
#         self.gene_features = nn.Sequential(
#             # nn.Linear(snp_number, 768),
#             # nn.BatchNorm1d(768),
#             # nn.Linear(snp_number, 384),
#             # nn.BatchNorm1d(384),
#             nn.Linear(snp_number, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )
#
#         self.feature_fusion1 = nn.Sequential(
#             # nn.BatchNorm1d(768 + snp_number),
#             # nn.ReLU(),
#             nn.Linear(768 + snp_number, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#         self.feature_fusion2 = nn.Sequential(
#             # 融合前再归一化
#             # nn.BatchNorm1d(768 * 2),
#             # nn.ReLU(),
#             # MLP
#             # nn.Linear(768 * 2, 384),
#             # nn.Linear(768 + 384, 128),
#             # nn.BatchNorm1d(128),
#             # nn.ReLU(),
#             # nn.Linear(128, 1),
#             # MLP2
#             nn.Linear(768 + 64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#         )
#         self.feature_fusion3 = nn.Sequential(
#             # nn.BatchNorm1d(768 * 2),
#             # nn.ReLU(),
#
#             # nn.TransformerEncoder(
#             # nn.TransformerEncoderLayer(d_model=768 * 2, nhead=4),
#             # num_layers=2
#             # ),
#             # nn.BatchNorm1d(768 * 2),
#             # nn.ReLU(),
#             # nn.Linear(768 * 2, 1),
#             nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(d_model=768 + 384, nhead=4),
#                 num_layers=2
#             ),
#             nn.BatchNorm1d(768 + 384),
#             nn.ReLU(),
#             nn.Linear(768 + 384, 1),
#         )
#
#         self.feature_fusion4 = nn.Sequential(
#             nn.MultiheadAttention(768 + 384, 4),
#             nn.BatchNorm1d(768 + 384),
#             nn.ReLU(),
#             nn.Linear(768 + 384, 1),
#         )
#
#         # 模型参数初始化
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.trunc_normal_(m.weight, std=0.2)
#             nn.init.constant_(m.bias, 0)
#
#     def forward(self, snps, image):
#         if self.snp_number <= 768:
#             gene_features = snps
#             feature_fusion = self.feature_fusion1
#         else:
#             gene_features = self.gene_features(snps)
#             feature_fusion = self.feature_fusion2
#
#         image_features = self.image_features(image)
#         x = torch.cat([image_features, gene_features], dim=1)
#         y = feature_fusion(x)
#         return y


nets = {
    f'SNPNet': SNPNet,
    f'ImageNet': ImageNet,
    f'SNPImageNet': SNPImageNet,
}


def nets_fake_data(device, model_name: str, batch_size: int, snp_numbers: int = 0):
    fake_data = {
        f'SNPNet': torch.randn(batch_size, snp_numbers).to(device),
        f'ImageNet': torch.randn(batch_size, 3, 224, 224).to(device),
        f'SNPImageNet': (
            torch.randn(batch_size, snp_numbers).to(device),
            torch.randn(batch_size, 3, 224, 224).to(device)),
    }
    if model_name not in fake_data.keys():
        raise ValueError(f'No such model: {model_name}')
    return fake_data[model_name]


__all__ = ['nets', 'nets_fake_data']
