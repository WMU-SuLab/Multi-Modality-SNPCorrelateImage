# -*- encoding: utf-8 -*-
"""
@File Name      :   count_students.py   
@Create Time    :   2024/1/6 19:48
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

import os

import pandas as pd

students_label_file_path = r"D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students.csv"
students_label_df = pd.read_csv(students_label_file_path)
students_ids = students_label_df['学籍号'].astype(str).tolist()

images_dir_path = r"D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\image\SuLabCohort\all"
image_file_names = [file_name for file_name in os.listdir(images_dir_path) if file_name.endswith('.jpg')]
image_ids = [file_name.split('_')[0] for file_name in image_file_names]

ids = [ID for ID in image_ids if ID in students_ids]
print(len(ids))
