# -*- encoding: utf-8 -*-
"""
@File Name      :   workflow.py
@Create Time    :   2023/3/4 20:03
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
import tqdm
from torch.cuda.amp import autocast

from .compute.metrics import count_metrics_binary_classification
from .multi_gpus import barrier, reduce_value
from .records import test_metrics_record
from .task import binary_classification_task, multi_classification_task


def train_valid_workflow(device, net, criterion, optimizer,  data_loader_iter, data_loaders, phase, scaler=None,
                         multi_gpu: bool = False):
    if phase == 'train':
        # 训练
        net.train()
    else:
        # 验证
        net.eval()
    y_true, y_pred, y_score = [], [], []
    running_loss = 0.0
    # 循环所有数据
    for inputs, labels in data_loader_iter:
        inputs = [each_input.to(device) for each_input in inputs]
        labels = labels.to(device)
        # 不知道是不是运算符重载的问题，必须用a+=b，a=a+b会报错
        # 也可以用torch.cat(a,b)，但是最后还需要再转一次为list
        # 这两种方法再最后转list之前得到的数据形式是不同，前者是list(tensor)，后者是tensor(list)，只是后者是一维的，所以转list之后看起来是一样的
        y_true += labels.int().reshape(-1).tolist()
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        with torch.set_grad_enabled(phase == 'train'):
            with autocast():
                outputs = net(*inputs)
                loss, y_pred_batch, y_score_batch = binary_classification_task(outputs, labels, criterion=criterion)
                # loss, y_pred_batch, y_score_batch = multi_classification_task(outputs, labels, criterion=criterion)
            # 只有训练的时候才会更新梯度
            if phase == 'train':
                if not scaler:
                    loss.backward()
                    optimizer.step()
                else:
                    # Scales loss，为了梯度放大
                    scaler.scale(loss).backward()
                    # scaler.step() 首先把梯度的值unscale回来.
                    # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                    # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                    scaler.step(optimizer)
                    # 准备着，看是否要增大scaler
                    scaler.update()
        # 计算损失
        if multi_gpu:
            barrier()
            loss = reduce_value(loss)
        y_pred += y_pred_batch
        y_score += y_score_batch
        running_loss += loss.item()
    # 计算损失
    epoch_loss = running_loss / len(data_loaders[phase].dataset)
    # 计算指标
    all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
    return epoch_loss, all_metrics


def test_workflow(device, net, data_loaders, writer):
    net.eval()
    # 注意不能使用连续赋值
    y_true, y_pred, y_score = [], [], []
    # 循环所有数据
    for inputs, labels in tqdm.tqdm(data_loaders):
        inputs = [each_input.to(device) for each_input in inputs]
        labels = labels.to(device)
        y_true += labels.int().reshape(-1).tolist()
        # 前向传播
        with torch.no_grad():
            outputs = net(*inputs)
            y_pred_batch, y_score_batch = binary_classification_task(outputs, labels)
            # y_pred_batch, y_score_batch = multi_classification_task(outputs, labels)
            # 不更新梯度
            y_pred += y_pred_batch
            y_score += y_score_batch
    # 记录指标
    test_metrics_record(y_true, y_pred, y_score, writer)


workflows = {
    'train': train_valid_workflow,
    'valid': train_valid_workflow,
    'test': test_workflow,
}

__all__ = ['workflows']
