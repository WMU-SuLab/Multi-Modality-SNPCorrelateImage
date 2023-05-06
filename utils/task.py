# -*- encoding: utf-8 -*-
"""
@File Name      :   task.py
@Create Time    :   2023/4/10 10:25
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


def binary_classification_task(outputs, labels, model_with_sigmoid=False, criterion=None, criterion_with_sigmoid=True):
    if model_with_sigmoid:
        y_pred = outputs.gt(0.5).int().reshape(-1).tolist()
        y_score = outputs.reshape(-1).tolist()
    else:
        new_outputs = torch.sigmoid(outputs)
        y_pred = new_outputs.gt(0.5).int().reshape(-1).tolist()
        y_score = new_outputs.reshape(-1).tolist()
    if criterion:
        if criterion_with_sigmoid and not model_with_sigmoid:
            loss = criterion(outputs, labels.float())
        elif criterion_with_sigmoid and model_with_sigmoid:
            raise ValueError('criterion and model cant both with sigmoid')
        elif not criterion_with_sigmoid and not model_with_sigmoid:
            raise ValueError('criterion and model cant both without sigmoid')
        else:
            new_outputs = torch.sigmoid(outputs)
            loss = criterion(new_outputs, labels.float())
        return loss, y_pred, y_score
    else:
        return y_pred, y_score


def multi_classification_task(outputs, labels, model_with_softmax=False, criterion=None,
                              criterion_with_softmax=True):
    if model_with_softmax:
        predicts_value, predicts_index = torch.max(outputs, dim=-1)
        y_pred = predicts_index.tolist()
        y_score = outputs
    else:
        new_outputs = torch.softmax(outputs, dim=-1)
        predicts_value, predicts_index = torch.max(new_outputs, dim=-1)
        y_pred = predicts_index.tolist()
        y_score = outputs
    if criterion:
        if criterion_with_softmax and not model_with_softmax:
            loss = criterion(outputs, labels)
        elif criterion_with_softmax and model_with_softmax:
            raise ValueError('criterion and model cant both with softmax')
        elif not criterion_with_softmax and not model_with_softmax:
            raise ValueError('criterion and model cant both without softmax')
        else:
            new_outputs = torch.softmax(outputs, dim=-1)
            loss = criterion(new_outputs, labels)
        return loss, y_pred, y_score
    else:
        return y_pred, y_score
