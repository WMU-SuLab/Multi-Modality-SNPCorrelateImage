# -*- encoding: utf-8 -*-
"""
@File Name      :   MultiModal.py   
@Create Time    :   2024/1/22 15:26
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

import torch
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from .Attention import SelfAttention, CrossAttention


class BaseMultiModalNet(nn.Module):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64):
        super(BaseMultiModalNet, self).__init__()
        # 基因模块模型
        self.gene_features = nn.Sequential(
            nn.Linear(snp_number, gene_features_num, bias=False),
            # nn.Linear(snp_number, gene_features_num),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        # 影像模块模型
        image_model = convnext_tiny(num_classes=1)
        # image_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        del image_model.classifier[-1]
        self.image_features = nn.Sequential(
            image_model,
            nn.BatchNorm1d(image_features_num),
            nn.ReLU(),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ConcatMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(ConcatMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.concat_fusion = nn.Sequential(
            nn.Linear(image_features_num + gene_features_num, fusion_features_num, bias=False),
            # nn.Linear(image_features_num + gene_features_num, fusion_features_num),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    # def apply(self, fn):
    #     self.gene_features.apply(fn)
    #     self.concat_fusion.apply(fn)
    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        x = torch.cat([image_feature, gene_feature], dim=1)
        y = self.concat_fusion(x)
        return y


class SelfAttentionMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(SelfAttentionMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features=nn.Sequential(
            nn.Linear(image_features_num , gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.self_attention_fusion = nn.Sequential(
            SelfAttention(gene_features_num + gene_features_num),
            nn.Linear(gene_features_num + gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature=self.image_gene_features(image_feature)
        x = torch.cat([image_gene_feature, gene_feature], dim=1)
        y = self.self_attention_fusion(x)
        return y


class MultiHeadAttentionMultiModalNet(BaseMultiModalNet):

    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(MultiHeadAttentionMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features = nn.Sequential(
            nn.Linear(image_features_num, gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.multi_head_attention = nn.MultiheadAttention(gene_features_num + gene_features_num, 4)
        self.multi_head_fusion = nn.Sequential(
            nn.Linear(gene_features_num + gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature = self.image_gene_features(image_feature)
        x = torch.cat([image_gene_feature, gene_feature], dim=1)
        y = self.multi_head_attention(x, x, x)[0]
        y = self.multi_head_fusion(y)
        return y


class TransformerMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(TransformerMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features = nn.Sequential(
            nn.Linear(image_features_num, gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.transformer_fusion = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=gene_features_num + gene_features_num, nhead=4),
                num_layers=2
            ),
            nn.Linear(gene_features_num + gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature = self.image_gene_features(image_feature)
        x = torch.cat([image_gene_feature, gene_feature], dim=1)
        y = self.transformer_fusion(x)
        return y


class CrossAttentionMultiModalNet(BaseMultiModalNet):
    def __init__(self, snp_number: int, image_features_num: int = 768, gene_features_num: int = 64,
                 fusion_features_num: int = 32):
        super(CrossAttentionMultiModalNet, self).__init__(snp_number, image_features_num, gene_features_num)
        self.image_gene_features = nn.Sequential(
            nn.Linear(image_features_num, gene_features_num, bias=False),
            nn.BatchNorm1d(gene_features_num),
            nn.ReLU(),
        )
        self.cross_attention = CrossAttention(gene_features_num, gene_features_num, gene_features_num)
        self.cross_attention_fusion = nn.Sequential(
            nn.Linear(gene_features_num, fusion_features_num, bias=False),
            nn.BatchNorm1d(fusion_features_num),
            nn.ReLU(),
            nn.Linear(fusion_features_num, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    def forward(self, snps, image):
        image_feature = self.image_features(image)
        gene_feature = self.gene_features(snps)
        image_gene_feature = self.image_gene_features(image_feature)
        x = self.cross_attention(gene_feature, image_gene_feature)
        # x=self.cross_attention(gene_feature, image_feature)
        y = self.cross_attention_fusion(x)
        return y
