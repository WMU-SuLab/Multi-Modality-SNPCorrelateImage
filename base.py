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
work_dirs_dir_name = 'work_dirs'
work_dirs_dir = os.path.join(project_dir_path, work_dirs_dir_name)

data_dir_name = 'data'
data_dir = os.path.join(work_dirs_dir, data_dir_name)
# data_dir = os.path.join(project_dir_path, data_dir_name)
data_divide_dir_name = os.path.join(data_dir_name, 'divide')
data_divide_dir = os.path.join(work_dirs_dir, data_divide_dir_name)
# data_divide_dir = os.path.join(project_dir_path, data_divide_dir_name)

records_dir_name = 'records'
records_dir = os.path.join(work_dirs_dir, records_dir_name)
weights_dir_name = 'weights'
weights_dir = os.path.join(records_dir, weights_dir_name)
checkpoints_dir_name = 'checkpoints'
checkpoints_dir = os.path.join(records_dir, checkpoints_dir_name)
logs_dir_name = 'logs'
logs_dir = os.path.join(records_dir, logs_dir_name)

outputs_dir_name = 'outputs'
outputs_dir = os.path.join(work_dirs_dir, outputs_dir_name)
results_dir_name = 'results'
results_dir = os.path.join(work_dirs_dir, results_dir_name)

root_dir_paths = [
    # work_dirs 文件夹，用于存放所有代码、文档的内容（包括数据、测试、模型权重等）
    work_dirs_dir,
    # data 文件夹，用于存放数据集
    data_dir,
    # data_divide 文件夹，用于存放划分好的数据集
    data_divide_dir,
    # records 文件夹，用于存放所有训练的记录
    records_dir,
    # weights 文件夹，用于存放模型的各项权重，一般用于预训练模型
    weights_dir,
    # checkpoints 文件夹，用于存放模型的各项参数文件，除了weights可能还有指标、轮数等等
    checkpoints_dir,
    # tensorboard 日志文件夹
    logs_dir,
    # outputs 文件夹，用于存放模型预测结果
    outputs_dir,
    # # results 文件夹，用于存放训练过程中模型各项参数的变化和临时结果
    # results_dir,
]

# for dir_path in root_dir_paths:
#     print(dir_path)
