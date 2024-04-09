# -*- encoding: utf-8 -*-
"""
@File Name      :   init.py
@Create Time    :   2022/11/29 11:28
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

import pandas as pd

from base import data_dir, all_participants_file_path

# 处理总的数据
raw_file_name = 'participants_label.txt'
raw_file_path = os.path.join(data_dir, raw_file_name)

# 读取数据
df = pd.read_csv(raw_file_path, sep='\t')
df = df.rename(columns={'eid': 'ID'})
df = df.fillna(0)

df['AMD'] = df['AMD'].apply(lambda x: 1 if x else 0)
df['DR'] = df['DR'].apply(lambda x: 1 if x else 0)
df['GLC'] = df['GLC'].apply(lambda x: 1 if x else 0)
df['RD'] = df['RD'].apply(lambda x: 1 if x else 0)

# 保存数据
df.to_csv(all_participants_file_path, index=False)
