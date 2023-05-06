# -*- encoding: utf-8 -*-
"""
@File Name      :   init.py
@Create Time    :   2023/4/20 21:55
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

import torch
from torch import nn, optim

from models import nets
from utils.early_stopping import LossEarlyStopping


def init_net(
        device, model_name: str, snp_numbers: int,
        pretrain_wts_path: str = None, pretrain_image_feature_checkpoint_path: str = None):
    # 模型，用net变量防止与torch的model冲突
    if Net := nets.get(model_name, None):
        net = Net(snp_numbers) if snp_numbers else Net()
    else:
        raise Exception(f"model_name {model_name} not found")
    net.to(device)
    if pretrain_wts_path and os.path.exists(pretrain_wts_path) and os.path.isfile(pretrain_wts_path):
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        missing_keys, unexpected_keys = net.load_state_dict(
            torch.load(pretrain_wts_path, map_location=device), strict=False)
    if pretrain_image_feature_checkpoint_path and os.path.exists(pretrain_image_feature_checkpoint_path) \
            and os.path.isfile(pretrain_image_feature_checkpoint_path):
        missing_keys, unexpected_keys = net.load_image_feature_state_dict(
            torch.load(pretrain_image_feature_checkpoint_path, map_location=device))
    return net


def init_strategy(net, lr, step_size, gamma, early_stopping_step, early_stopping_delta):
    # 定义损失函数和优化器
    optimizer = optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()), lr=lr)
    # 学习率每 n 个 epoch 减少一次，衰减为原来的十分之一
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # 确定损失函数
    criterion = nn.BCEWithLogitsLoss()
    # 早停策略
    loss_early_stopping = LossEarlyStopping(patience=early_stopping_step, delta=early_stopping_delta)
    return optimizer, scheduler, criterion, loss_early_stopping
