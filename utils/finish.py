# -*- encoding: utf-8 -*-
"""
@File Name      :   finish.py
@Create Time    :   2023/4/20 15:19
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

import time

from .workflow import workflows


def finish_train(device, net, data_loaders, writer, best_f1, best_model_wts, since):
    # 训练全部完成
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f}'.format(best_f1))
    # 加载最佳模型权重，最后做一次总的测试
    net.load_state_dict(best_model_wts)
    workflows['test'](device, net, data_loaders['valid'], writer)
