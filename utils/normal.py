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
    return (x - x.mean()) / x.std()


# 归一化
def min_max_normalization(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())
