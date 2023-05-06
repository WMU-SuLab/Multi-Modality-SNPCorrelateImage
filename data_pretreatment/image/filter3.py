# -*- encoding: utf-8 -*-
"""
@File Name      :   filter3.py
@Create Time    :   2023/4/13 10:30
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

# 根据双眼情况定义的高度近视人群筛选图片
# 因为有的人拍了多张图片，所以要随机选一张
df = pd.read_csv(
    r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_two_eyes_merge.csv',
    dtype={'学籍号': str})
student_ids = df['学籍号'].tolist()
print(len(student_ids))
file_names = defaultdict(list)
for file_name in os.listdir(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc'):
    file_names['_'.join(file_name.split('_')[0:2])].append(file_name)
print(len(file_names.keys()))
count = 0
for student_id in student_ids:
    # if (OSs := file_names[f'{student_id}_OS']) and (ODs := file_names[f'{student_id}_OD']):
    #     # print(student_id)
    #     count += 1
    #     random.shuffle(OSs)
    #     random.shuffle(ODs)
    #     os_file = random.choice(OSs)
    #     od_file = random.choice(ODs)
    #     copy2(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc\{}'.format(os_file),
    #           r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc_high_myopia_snp\{}'.format(os_file))
    #     copy2(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc\{}'.format(od_file),
    #           r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc_high_myopia_snp\{}'.format(od_file))
    if OSs := file_names[f'{student_id}_OS']:
        random.shuffle(OSs)
        os_file = random.choice(OSs)
        copy2(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc\{}'.format(os_file),
              r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc_high_myopia_snp\{}'.format(os_file))
    if ODs := file_names[f'{student_id}_OD']:
        random.shuffle(ODs)
        od_file = random.choice(ODs)
        copy2(r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc\{}'.format(od_file),
              r'D:\BaiduSyncdisk\Data\SuLabCohort\image\students_qc_high_myopia_snp\{}'.format(od_file))
print(count)
