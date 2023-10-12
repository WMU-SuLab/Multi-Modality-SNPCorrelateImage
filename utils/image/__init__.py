# -*- encoding: utf-8 -*-
"""
@File Name      :   __init__.py.py   
@Create Time    :   2023/8/1 20:07
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@other information
"""
__auth__ = 'diklios'

import numpy as np
import torch


def tensor2numpy(tensor: torch.Tensor, mean: list[float, ...] = [0.485, 0.456, 0.406],
                 std: list[float, ...] = [0.229, 0.224, 0.225], ):
    image = tensor.to('cpu').clone().detach().numpy().squeeze()
    # 在torch中把颜色通道放到了第一个维度，在PIL中把颜色通道放到了最后一个维度，所以需要还原回去
    image = image.transpose((1, 2, 0))
    # 将图像的标准化处理还原回去
    image = image * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    return image
