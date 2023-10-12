# -*- encoding: utf-8 -*-
"""
@File Name      :   normal.py
@Create Time    :   2023/4/11 14:13
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


def normalization(x: torch.Tensor):
    """
    归一化，归一化到均值为0，标准差为1的分布
    :param x:
    :return:
    """
    """
    :param x: 
    :return: 
    """
    return (x - x.mean()) / x.std(0)


# 归一化
def min_max_normalization(x: torch.Tensor):
    """
    归一化，归一化到[0,1]
    :param x:
    :return:
    """
    return (x - x.min()) / (x.max() - x.min())


def negative_normalization(x: torch.Tensor):
    """
    归一化，归一化到[-1,1]
    :param x:
    :return:
    """
    normalized = (x - x.min()) / (x.max() - x.min())
    return 2 * normalized - 1
