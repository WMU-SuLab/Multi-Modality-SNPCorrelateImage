# -*- encoding: utf-8 -*-
"""
@File Name      :   base.py
@Create Time    :   2022/11/2 10:28
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

import os

project_dir_path = os.path.dirname(__file__)
data_dir_name = 'data'
data_dir = os.path.join(project_dir_path, data_dir_name)
data_divide_dir_name = os.path.join('data', 'divide')
data_divide_dir = os.path.join(project_dir_path, data_divide_dir_name)
weights_dir_name = 'weights'
weights_dir = os.path.join(project_dir_path, weights_dir_name)
checkpoints_dir_name = 'checkpoints'
checkpoints_dir = os.path.join(project_dir_path, checkpoints_dir_name)
logs_dir_name = 'logs'
logs_dir = os.path.join(project_dir_path, logs_dir_name)
outputs_dir_name = 'outputs'
outputs_dir = os.path.join(project_dir_path, outputs_dir_name)
work_dirs_dir_name = 'work_dirs'
work_dirs_dir = os.path.join(project_dir_path, work_dirs_dir_name)
results_dir_name = 'results'
results_dir = os.path.join(project_dir_path, results_dir_name)

root_dir_paths = [
    # data 文件夹，用于存放数据集
    data_dir,
    # data_divide 文件夹，用于存放划分好的数据集
    data_divide_dir,
    # weights 文件夹，用于存放模型的各项权重，一般用于预训练模型
    weights_dir,
    # checkpoints 文件夹，用于存放模型的各项参数文件，除了weights可能还有指标、轮数等等
    checkpoints_dir,
    # tensorboard 日志文件夹
    logs_dir,
    # outputs 文件夹，用于存放模型结果
    outputs_dir,
    # # work_dirs 文件夹，用于存放训练结果及趣味 demo 输出结果
    # work_dirs_dir,
    # # results 文件夹，用于存放训练过程中模型各项参数的变化和临时结果
    # results_dir,
]
