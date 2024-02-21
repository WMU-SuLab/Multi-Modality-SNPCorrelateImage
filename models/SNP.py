# -*- encoding: utf-8 -*-
"""
@File Name      :   SNPNet.py   
@Create Time    :   2024/1/22 14:19
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

import math

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .performer_pytorch import Performer
from torch import nn
from torch.autograd import Variable


class SimpleLinear(nn.Module):
    def __init__(self, snp_number: int):
        super(SimpleLinear, self).__init__()
        self.snp_number = snp_number
        # hide_layer_num = self.count_hide_layer_num()
        self.snp_mlp = nn.Sequential(
            nn.Linear(snp_number, 64),
            # nn.Dropout(0.2),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Linear(snp_number, hide_layer_num),
            # nn.BatchNorm1d(hide_layer_num),
            # nn.ReLU(),
            # nn.Linear(hide_layer_num, 1),
        )

    def count_hide_layer_num(self):
        stop = False
        count = 0
        while not stop:
            count += 1
            if math.pow(2, math.pow(2, count)) <= self.snp_number < math.pow(2, math.pow(2, count + 1)):
                stop = True
        return int(math.pow(2, count + 1))

    def forward(self, x):
        x = self.snp_mlp(x)
        return x


class DoubleLayerSimpleMLP(nn.Module):
    def __init__(self, snp_number: int):
        super(DoubleLayerSimpleMLP, self).__init__()
        self.snp_number = snp_number
        self.snp_mlp = nn.Sequential(
            nn.Linear(snp_number, 512),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.snp_mlp(x)
        return x


class BaseMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(BaseMLPBlock, self).__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.mlp_block(x)
        return x


class ResMLP(nn.Module):
    def __init__(self, snp_number: int):
        super(ResMLP, self).__init__()
        self.mlp_block1 = BaseMLPBlock(in_features=snp_number, out_features=64)
        self.mlp_block2 = BaseMLPBlock(in_features=64, out_features=64)
        self.mlp = nn.Sequential(
            # nn.Linear(in_features=64, out_features=8),
            # nn.BatchNorm1d(8),
            # nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        x1 = self.mlp_block1(x)
        x2 = self.mlp_block2(x1)
        x3 = x1 + x2
        y = self.mlp(x3)
        return y


neurons = 150
primary_capslen = 4
digital_capslen = 16
ks = 5
stride = 2
filters = 32
num_iterations = 3


class ConvCaps2D(nn.Module):
    def __init__(self):
        super(ConvCaps2D, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=primary_capslen,
                                                 kernel_size=(ks, 1), stride=stride) for _ in range(filters)])

    def squash(self, tensor, dim=-1):
        norm = (tensor ** 2).sum(dim=dim, keepdim=True)  # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm)  # scale.size()  is (None, 1152, 1)
        return scale * tensor / torch.sqrt(norm)

    def forward(self, x):
        # print(x.size())
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 1)
        outputs = [capsule(x).view(x.size(0), primary_capslen, -1) for capsule in
                   self.capsules]  # 32 list of (None, 1, 8, 36)
        # print(len(outputs))#32
        # print(len(outputs[0]))#32
        # print(len(outputs[0][0]))#4
        # print(len(outputs[0][0][0]))#73   32 list of (None,1,4,73)
        # 拼接＋转置
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)  # outputs.size() is (None, 1152, 8)
        # print(outputs.shape)#(32,2336,4)
        return self.squash(outputs)


class Caps1D(nn.Module):
    def __init__(self):
        super(Caps1D, self).__init__()
        self.num_iterations = num_iterations
        self.num_caps = 2  # equals to class number
        self.num_routes = (int((neurons - ks) / stride) + 1) * filters
        self.in_channels = primary_capslen
        self.out_channels = digital_capslen

        self.W = nn.Parameter(torch.randn(self.num_caps, self.num_routes, self.in_channels,
                                          self.out_channels))  # class,weight,len_capsule,capsule_layer

    #         self.W = nn.Parameter(torch.randn(3, 3136, 8, 32)) # num_caps, num_routes, in_channels, out_channels
    # (2,2336,4,32)

    def softmax(self, x, dim=1):
        transposed_input = x.transpose(dim, len(x.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)

    def squash(self, tensor, dim=-1):
        norm = (tensor ** 2).sum(dim=dim, keepdim=True)  # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm)
        return scale * tensor / torch.sqrt(norm)

    # Routing algorithm
    def forward(self, u):
        # u.size() is (None, 1152, 8)
        '''
        From documentation
        For example, if tensor1 is a j x 1 x n x m Tensor and tensor2 is a k x m x p Tensor,
        out will be an j x k x n x p Tensor.

        We need j = None, 1, n = 1152, k = 10, m = 8, p = 16
        '''
        # print(u.size())#torch.Size([32, 2336, 4])
        u_ji = torch.matmul(u[:, None, :, None, :], self.W)  # u_ji.size() is (None, 10, 1152, 1, 16)
        # print(u_ji.size())#矩阵乘法 torch.Size([32, 2, 2336, 1, 16])
        b = Variable(torch.zeros(u_ji.size()))  # b.size() is (None, 10, 1152, 1, 16)
        # print(b.size())#torch.Size([32, 2, 2336, 1, 16])
        # b = b.to(device) # using gpu
        b = b.cuda()

        for i in range(self.num_iterations):
            c = self.softmax(b, dim=2)
            v = self.squash((c * u_ji).sum(dim=2, keepdim=True))  # v.size() is (None, 10, 1, 1, 16)
            # print(v.size())#torch.Size([32, 2, 1, 1, 16])

            if i != self.num_iterations - 1:
                delta_b = (u_ji * v).sum(dim=-1, keepdim=True)
                b = b + delta_b

        # Now we simply compute the length of the vectors and take the softmax to get probability.
        v = v.squeeze()
        classes = (v ** 2).sum(dim=-1) ** 0.5
        # classes = F.softmax(classes)

        return classes


class CapsuleNet(nn.Module):
    def __init__(self, snp_number):
        super().__init__()
        self.snp_number = snp_number
        self.gene_mlp = nn.Sequential(
            nn.Linear(snp_number, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            ConvCaps2D(),
            Caps1D()
        )

    def forward(self, snps):
        x = self.gene_mlp(snps)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length):
        super(PositionalEncoding, self).__init__()
        self.length = length
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(1, self.length, d_model)
        position = torch.arange(0, self.length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Calculate the positional encoding for each position
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        print(pe.shape)
        # Resize the positional encoding to match the height and width
        # pe = pe[:self.height * self.width, :].view(self.height, self.width, -1).transpose(0, 2).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return self.dropout(x)


class DeepExGRS(nn.Module):
    def __init__(self, snp_number):
        super().__init__()
        self.snp_number = snp_number

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (l p1) -> b (l) (p1 c)', p1=768),
            nn.Linear(768, 192),
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, 192))
        # todo:余弦位置编码
        self.cosine_pos_embed = PositionalEncoding(192, math.ceil(snp_number / 192) + 1)
        # todo:自学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, math.ceil(snp_number / 192) + 1, 192))
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=192, nhead=4),
        #     num_layers=2, norm=nn.LayerNorm(192),
        # )
        self.transformer = Performer(
            dim=192,
            depth=3,
            heads=8,
        )
        self.mlp = nn.Sequential(
            nn.Linear(192, 192),
            nn.GELU(),
            nn.Linear(192, 1)
        )

    def forward(self, snps):
        # snps = snps.long()
        snps += 1
        x = F.pad(snps, (1, 192 - 1 - (self.snp_number % 192)), 'constant', 0)
        # x = F.one_hot(x, num_classes=4)
        # todo:权重
        # x = x.transpose(1, 2).contiguous().float()
        x = x.unsqueeze(1)
        x = self.to_patch_embedding(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.cosine_pos_embed(x)
        # x = x + self.pos_embed
        x = self.transformer(x)
        x0 = x[:, 0]
        y = self.mlp(x0)
        return y
