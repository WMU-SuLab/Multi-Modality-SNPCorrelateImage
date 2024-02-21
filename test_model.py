# -*- encoding: utf-8 -*-
"""
@File Name      :   test_model.py   
@Create Time    :   2024/2/6 9:45
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
from models import SNPImageNet
import torch
gene=torch.randn(2, 2)
image=torch.randn(2, 3, 224, 224)
a=SNPImageNet(2)
a(gene, image)