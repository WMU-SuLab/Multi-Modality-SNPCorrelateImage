# -*- encoding: utf-8 -*-
"""
@File Name      :   encode.py
@Create Time    :   2023/4/10 11:20
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


def classifications_convert_to_one_hot(classifications, num_classes):
    """
    将分类结果转换为one-hot编码
    :param classifications: 分类结果，从0开始
    :param num_classes: 分类数
    :return: one-hot编码
    """
    one_hot = torch.zeros(classifications.size(0), num_classes)
    one_hot.scatter_(1, classifications.view(-1, 1), 1)
    return one_hot
