# -*- encoding: utf-8 -*-
"""
@File Name      :   glc.py
@Create Time    :   2023/5/22 15:29
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
from datetime import datetime
from shutil import copy2

import pandas as pd

suffix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# 提取标签数据（诊断和问卷重合部分）
glc_q_df = pd.read_csv('/share2/pub/biobank/phenotype/which_eye_affected.csv', dtype={'eid': str})
glc_q_df = glc_q_df[['eid', '6119-0.0', '6119-1.0']]
glc_q_df = glc_q_df[glc_q_df['6119-0.0'].notnull() | glc_q_df['6119-1.0'].notnull()]
glc_q_ids = glc_q_df['eid'].tolist()

glc_diagnose_df = pd.read_csv('/share2/home/sunhj/GLC_phenotype.txt', names=['id', 'id2', 'label'], sep='\t',
                              dtype={'id': str, 'id2': str, 'label': int})
glc_diagnose_ids = glc_diagnose_df[glc_diagnose_df['label'] == 1]['id'].tolist()

glc_ids = list(set(glc_q_ids) & set(glc_diagnose_ids))
df = glc_q_df[glc_q_df['eid'].isin(glc_ids)]

# 提取图片数据
dir_path = os.path.dirname(__file__)
glc_image_dir_name = f'glc_images{suffix}'
glc_image_dir_path = os.path.join(dir_path, glc_image_dir_name)
if not os.path.exists(glc_image_dir_path):
    os.mkdir(glc_image_dir_path)

image_paths_dict = {}
image_left_dir_path = '/share2/pub/biobank/Image/fundus_image_left'
image_paths_left = []
for image_name in os.listdir(image_left_dir_path):
    if image_name.endswith('.png'):
        image_name_split = image_name.split('.')[0].split('_')
        eid = image_name_split[0]
        instance = image_name_split[2]
        count = image_name_split[3]
        image_paths_left.append(eid)
        image_paths_dict[f'{eid}-{instance}_OS_{count}'] = image_name
glc_left_set = set(glc_ids) & set(image_paths_left)
# glc_left_set_len = len(glc_left_set)
# print(glc_left_set_len)

image_right_dir_path = '/share2/pub/biobank/Image/fundus_image_right'
image_paths_right = []
for image_name in os.listdir(image_right_dir_path):
    if image_name.endswith('.png'):
        image_name_split = image_name.split('.')[0].split('_')
        eid = image_name_split[0]
        instance = image_name_split[2]
        count = image_name_split[3]
        image_paths_right.append(eid)
        image_paths_dict[f'{eid}-{instance}_OD_{count}'] = image_name
glc_right_set = set(glc_ids) & set(image_paths_right)
# glc_right_set_len = len(glc_right_set)
# print(glc_right_set_len)


new_df = pd.DataFrame(columns=['eid', 'OS_GLC', 'OD_GLC'])
for index, row in df.iterrows():
    eid = row['eid']
    for instance in range(2):
        exist = False
        glc = row[f'6119-{instance}.0']
        if glc == 1:
            new_series = pd.Series({'eid': f"{eid}-{instance}", 'OS_GLC': 0, 'OD_GLC': 1})
        elif glc == 2:
            new_series = pd.Series({'eid': f"{eid}-{instance}", 'OS_GLC': 1, 'OD_GLC': 0})
        elif glc == 3:
            new_series = pd.Series({'eid': f"{eid}-{instance}", 'OS_GLC': 1, 'OD_GLC': 1})
        else:
            new_series = pd.Series({'eid': f"{eid}-{instance}", 'OS_GLC': 0, 'OD_GLC': 0})
        for count in range(2):
            OS = f"{eid}-{instance}_OS_{count}"
            OD = f"{eid}-{instance}_OD_{count}"
            if image_paths_dict.get(OS, None):
                exist = True
                copy2(os.path.join(image_left_dir_path, image_paths_dict[OS]),
                      os.path.join(glc_image_dir_path, f'{OS}.png'))
            if image_paths_dict.get(OD, None):
                exist = True
                copy2(os.path.join(image_right_dir_path, image_paths_dict[OD]),
                      os.path.join(glc_image_dir_path, f'{OD}.png'))
        if exist:
            new_df = pd.concat([new_df, new_series.to_frame().T], axis=0)
            # new_df=new_df.append(new_series, ignore_index=True)

# 提取非青光眼且非其他眼病的数据
glc_all_or_set = glc_left_set | glc_right_set
glc_all_or_set_len = len(glc_all_or_set)
print(glc_all_or_set_len)

disease_df = pd.read_csv('/share2/pub/biobank/Image/participants_label.txt', sep='\t', dtype={'eid': str})
disease_df = disease_df.fillna(0)
no_disease_df = disease_df[
    (disease_df['AMD'] == 0) &
    (disease_df['DR'] == 0) &
    (disease_df['GLC'] == 0) &
    (disease_df['RD'] == 0)
    ]
no_disease_ids = no_disease_df['eid'].tolist()
not_glc_ids_choice = random.choices(no_disease_ids, k=glc_all_or_set_len)
for eid in not_glc_ids_choice:
    for instance in range(2):
        exist = False
        for count in range(2):
            OS = f"{eid}-{instance}_OS_{count}"
            OD = f"{eid}-{instance}_OD_{count}"
            if image_paths_dict.get(OS, None):
                exist = True
                copy2(os.path.join(image_left_dir_path, image_paths_dict[OS]),
                      os.path.join(glc_image_dir_path, f'{OS}.png'))
            if image_paths_dict.get(OD, None):
                exist = True
                copy2(os.path.join(image_right_dir_path, image_paths_dict[OD]),
                      os.path.join(glc_image_dir_path, f'{OD}.png'))
        if exist:
            new_df = pd.concat([new_df, pd.Series({'eid': f"{eid}-{instance}", 'OS_GLC': 0, 'OD_GLC': 0}).to_frame().T],
                               axis=0)
            # new_df = new_df.append(pd.Series({'eid': f"{eid}-{instance}", 'OS_GLC': 0, 'OD_GLC': 0}), ignore_index=True)
# 导出
new_df.to_csv(os.path.join(dir_path, f'glc_{suffix}.csv'), index=False)
with open(os.path.join(dir_path, f'glc_{suffix}.txt'), 'w') as f:
    all_ids = new_df['eid'].str.split('-').str[0].tolist()
    f.write('\n'.join(list(set(all_ids))))
