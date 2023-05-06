# -*- encoding: utf-8 -*-
"""
@File Name      :   early_stopping.py
@Create Time    :   2023/4/14 15:57
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

import numpy as np


class LossEarlyStopping:
    """
    这个早停类不做除了是否早停外的其他操作（保存模型和权重等等）
    """
    def __init__(self, patience: int = 7, delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        patience：自上次模型在验证集上损失降低之后等待的时间，此处设置为7
        counter：计数器，当其值超过patience时候，使用early stopping
        best_score：记录模型评估的最好分数
        early_step：决定模型要不要early stop，为True则停
        val_loss_min：模型评估损失函数的最小值，默认为正无穷(np.Inf)
        delta：表示模型损失函数改进的最小值，当超过这个值时候表示模型有所改进
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'{self.__class__.__name__} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


class MetricEarlyStopping:
    """
    指标型的基本可以仿照loss型的写，但是要注意指标有越大越好和越小越好，注意增加这个接口
    """

    def __init__(self):
        pass
