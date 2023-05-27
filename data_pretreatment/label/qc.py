# -*- encoding: utf-8 -*-
"""
@File Name      :   qc.py
@Create Time    :   2023/4/25 16:21
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

from datetime import datetime

import pandas as pd

df = pd.read_excel(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\温医大学生结果导出-原始汇总0826.xlsx', sheet_name='Sheet1',
                   dtype={'学籍号': str, '条形码': str, '证件号': str})
df = df.rename(columns={'左眼眼轴长度(AL)': 'AL_OS', '右眼眼轴长度(AL)': 'AL_OD'})
print(df.shape)
df.to_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all.csv', index=False)
df_students = df[(df['学籍号'].str.len() == 10) | (df['学籍号'].str.len() == 11)]
print(df_students.shape)
df_students = df_students.dropna(subset=['条形码'])
print(df_students.shape)
df_students = df_students.drop_duplicates(subset=['条形码'], keep=False)
print(df_students.shape)
df_students.to_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students.csv',
                   index=False)
# df_students = df_students.dropna(subset=['证件号'])
# df_students = df_students[df_students['证件号'].str.len() == 18]
print(df_students.shape)
df_students_barcodes = df_students[['学生名称', '学籍号', '条形码']]
print(df_students_barcodes.shape)
df_students_barcodes.to_csv(
    r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_barcodes.csv', index=False)
# df_students['年龄'] = df_students['证件号'].apply(
#     lambda x: (datetime.now() - datetime.strptime(x.strip()[6:14], '%Y%m%d')).days)
df_qc = df_students[['学生名称', '学籍号', '条形码', '性别', '年龄', 'SE_OS', 'SE_OD', 'AL_OS', 'AL_OD']].dropna()
print(df_qc.shape)
df_qc.to_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc.csv',
             index=False)
