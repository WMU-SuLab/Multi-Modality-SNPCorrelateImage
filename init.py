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
        device, model_name: str, snp_number: int,
        pretrain_checkpoint_path: str = None, pretrain_image_feature_checkpoint_path: str = None):
    # 模型，用net变量防止与torch的model冲突
    if Net := nets.get(model_name, None):
        net = Net(snp_number) if snp_number else Net()
    else:
        raise Exception(f"model_name {model_name} not found")
    net.to(device)
    if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path) and os.path.isfile(pretrain_checkpoint_path):
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        # missing_keys, unexpected_keys = net.load_state_dict(
        #     torch.load(pretrain_checkpoint_path, map_location=device)['state_dict'], strict=True)
        missing_keys, unexpected_keys = net.load_state_dict(
            torch.load(pretrain_checkpoint_path, map_location=device)['model'], strict=True)
    if pretrain_image_feature_checkpoint_path and os.path.exists(pretrain_image_feature_checkpoint_path) \
            and os.path.isfile(pretrain_image_feature_checkpoint_path):
        missing_keys, unexpected_keys = net.load_image_feature_state_dict(
            torch.load(pretrain_image_feature_checkpoint_path, map_location=device))
    return net


def init_strategy(net, lr, step_size, gamma, last_epoch, pretrain_checkpoint_path, early_stopping_step,
                  early_stopping_delta):
    if last_epoch == -1:
        # 定义损失函数和优化器
        optimizer = optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()), lr=lr,
                               # weight_decay=5e-4
                               )
        # 学习率每 n 个 epoch 减少一次，衰减为原来的十分之一
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        checkpoint = torch.load(pretrain_checkpoint_path)
        optimizer = optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    # 确定损失函数
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # 早停策略
    loss_early_stopping = LossEarlyStopping(patience=early_stopping_step, delta=early_stopping_delta)
    return optimizer, scheduler, criterion, loss_early_stopping
