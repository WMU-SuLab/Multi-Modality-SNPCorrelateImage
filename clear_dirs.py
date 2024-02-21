# -*- encoding: utf-8 -*-
"""
@File Name      :   clear_dirs.py
@Create Time    :   2023/4/13 21:46
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
from shutil import rmtree

from base import logs_dir,  checkpoints_dir,code_dir

# weights_dirs = os.listdir(weights_dir)
checkpoints_dirs = os.listdir(checkpoints_dir)
logs_dirs = os.listdir(logs_dir)
code_dirs = os.listdir(code_dir)
# for dir_name in os.listdir(weights_dir):
#     if dir_name not in logs_dirs:
#         rmtree(os.path.join(weights_dir, dir_name), ignore_errors=True)
for dir_name in os.listdir(checkpoints_dir):
    if dir_name not in logs_dirs:
        rmtree(os.path.join(checkpoints_dir, dir_name), ignore_errors=True)
for dir_name in os.listdir(code_dir):
    if dir_name not in logs_dirs:
        rmtree(os.path.join(code_dir, dir_name), ignore_errors=True)
