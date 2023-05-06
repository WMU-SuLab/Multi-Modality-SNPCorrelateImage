# -*- encoding: utf-8 -*-
"""
@File Name      :   filter4.py
@Create Time    :   2023/4/15 15:15
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
import random
from collections import defaultdict
from shutil import copy2

import pandas as pd

# 根据单眼情况定义的高度近视人群筛选图片
# 因为有的人拍了多张图片，所以要随机选一张
df = pd.read_csv(
    r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc_one_eye_merge.csv',
    dtype={'学籍号': str})
print(df.shape)
file_names = defaultdict(list)
for file_name in os.listdir(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc'):
    file_names['_'.join(file_name.split('_')[0:2])].append(file_name)
print(len(file_names.keys()))
count = 0
for index, row in df.iterrows():
    if (OSs := file_names[f"{row['学籍号']}_OS"]) and (row['OS_high_myopia'] == 1 or row['OS_high_myopia'] == 0):
        count += 1
        random.shuffle(OSs)
        os_file = random.choice(OSs)
        copy2(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc\{}'.format(os_file),
              r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc_high_myopia\{}'.format(os_file))
    if (ODs := file_names[f"{row['学籍号']}_OD"]) and (row['OD_high_myopia'] == 1 or row['OD_high_myopia'] == 0):
        count += 1
        random.shuffle(ODs)
        od_file = random.choice(ODs)
        copy2(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc\{}'.format(od_file),
              r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc_high_myopia\{}'.format(od_file))
print(count)
