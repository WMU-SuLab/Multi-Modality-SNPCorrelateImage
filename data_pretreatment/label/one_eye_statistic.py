# -*- encoding: utf-8 -*-
"""
@File Name      :   one_eye_statistic.py
@Create Time    :   2023/4/15 15:18
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

datatime_now_str = datetime.now().strftime('%Y%m%d%H%M%S')
df = pd.read_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc.csv')
# high_myopia_OS_df = df[(df['SE_OS'] <= -6) & (df['AL_OS'] >= 25)]
high_myopia_OS_df = df[(df['SE_OS'] <= -6) & (df['AL_OS'] >= 26)]
print(high_myopia_OS_df.shape)
high_myopia_OS_df.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc_one_eye_high_myopia_OS_{datatime_now_str}.csv',
    index=False)
# high_myopia_OD_df = df[(df['SE_OD'] <= -6) & (df['AL_OD'] >= 25)]
high_myopia_OD_df = df[(df['SE_OD'] <= -6) & (df['AL_OD'] >= 26)]
print(high_myopia_OD_df.shape)
high_myopia_OD_df.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc_one_eye_high_myopia_OD_{datatime_now_str}.csv',
    index=False)
# not_high_myopia_OS_df = df[(-3 <= df['SE_OS']) & (df['SE_OS'] <= -0.5)]
not_high_myopia_OS_df = df[(-3 <= df['SE_OS']) & (df['SE_OS'] <= -0.5) & (df['AL_OS'] <= 24)]
print(not_high_myopia_OS_df.shape)
not_high_myopia_OS_df.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc_one_eye_not_high_myopia_OS_{datatime_now_str}.csv',
    index=False)
# not_high_myopia_OD_df = df[(-3 <= df['SE_OD']) & (df['SE_OD'] <= -0.5)]
not_high_myopia_OD_df = df[(-3 <= df['SE_OD']) & (df['SE_OD'] <= -0.5) & (df['AL_OD'] <= 24)]
print(not_high_myopia_OD_df.shape)
not_high_myopia_OD_df.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc_one_eye_not_high_myopia_OD_{datatime_now_str}.csv',
    index=False)
columns = df.columns
high_myopia_OS_df = high_myopia_OS_df[['学籍号']].copy()
high_myopia_OD_df = high_myopia_OD_df[['学籍号']].copy()
not_high_myopia_OS_df = not_high_myopia_OS_df[['学籍号']].copy()
not_high_myopia_OD_df = not_high_myopia_OD_df[['学籍号']].copy()
high_myopia_OS_df['OS_high_myopia'] = 1
high_myopia_OD_df['OD_high_myopia'] = 1
not_high_myopia_OS_df['OS_high_myopia'] = 0
not_high_myopia_OD_df['OD_high_myopia'] = 0
OS_df = pd.concat([high_myopia_OS_df, not_high_myopia_OS_df], axis=0)
OD_df = pd.concat([high_myopia_OD_df, not_high_myopia_OD_df], axis=0)
# high_myopia_df_merge = pd.merge(high_myopia_OS_df, high_myopia_OD_df, on='学籍号', how='outer')
# not_high_myopia_df_merge = pd.merge(not_high_myopia_OS_df, not_high_myopia_OD_df, on='学籍号', how='outer')
# df_merge = pd.concat([high_myopia_df_merge, not_high_myopia_df_merge], axis=0)
df_merge = pd.merge(OS_df, OD_df, on='学籍号', how='outer')
# -1代表既不是高度近视也不是轻度近视
df_merge = df_merge.fillna(-1)
df_merge.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc_one_eye_merge_{datatime_now_str}.csv',
    index=False)
