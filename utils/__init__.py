# -*- encoding: utf-8 -*-
"""
@File Name      :   __init__.py.py
@Create Time    :   2023/2/6 10:59
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

import random

import numpy as np
import torch


def setup_seed(seed: int):
    """
    保证每次实验的结果是一样的
    :param seed: 随机数种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(20230410)
