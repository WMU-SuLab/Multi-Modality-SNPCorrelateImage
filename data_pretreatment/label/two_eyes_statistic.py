# -*- encoding: utf-8 -*-
"""
@File Name      :   statistics.py
@Create Time    :   2023/4/12 11:12
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
df = pd.read_excel(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\温医大学生结果导出-原始汇总0826.xlsx', sheet_name='Sheet1',
                   dtype={'学籍号': str})
# print(df.head())
df = df[(df['学籍号'].str.len() == 11) | (df['学籍号'].str.len() == 10)]
df = df[['学生名称', '学籍号', '性别', 'SE_OS', 'SE_OD', '左眼眼轴长度(AL)', '右眼眼轴长度(AL)']]
df.rename({'左眼眼轴长度(AL)': 'AL_OS', '右眼眼轴长度(AL)': 'AL_OD'}, axis=1, inplace=True)
# df.to_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students.csv', index=False)
print(df.shape)
df = df.dropna()
print(df.shape)
# df.to_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc.csv', index=False)
# # 统计是否高度近视
# df['SE_OS-high_myopia'] = df['SE_OS'].apply(lambda x: 1 if x <= -6 else 0)
# df['SE_OD-high_myopia'] = df['SE_OD'].apply(lambda x: 1 if x <= -6 else 0)
# df['AL_OS-high_myopia'] = df['AL_OS'].apply(lambda x: 1 if x <= 26.5 else 0)
# df['AL_OD-high_myopia'] = df['AL_OD'].apply(lambda x: 1 if x <= 26.5 else 0)
# df['SE-high_myopia_strict'] = df['SE_OS-high_myopia'] & df['SE_OD-high_myopia']
# df['AL-high_myopia_strict'] = df['AL_OS-high_myopia'] & df['AL_OD-high_myopia']
# df['SE-high_myopia_lax'] = df['SE_OS-high_myopia'] | df['SE_OD-high_myopia']
# df['AL-high_myopia_lax'] = df['AL_OS-high_myopia'] | df['AL_OD-high_myopia']
# df['SE-high_myopia_lax2'] = df[['SE_OS', 'SE_OD']].apply(
#     lambda x: 1 if (x['SE_OS'] <= -6 and x['SE_OD'] <= -4) or (x['SE_OD'] <= -6 and x['SE_OS'] <= -4) else 0, axis=1)
# # 统计近视等级
# df['SE_OS-myopia_level'] = df['SE_OS'].apply(lambda x: 2 if x <= -6 else (1 if -6 <= x <= -3 else 0))
# df['SE_OD-myopia_level'] = df['SE_OD'].apply(lambda x: 2 if x <= -6 else (1 if -6 <= x <= -3 else 0))
# # 统计分布情况
# # 高度近视宽松标准
# df_high_myopia_lax = df[df['SE-high_myopia_lax'] == 1][['AL_OS', 'AL_OD']]
# df_high_myopia_lax_len = df_high_myopia_lax.shape[0]
# print(f'高度近视宽松标准（任意一只眼小于-6D）人数：{df_high_myopia_lax_len}')
# plt.figure()
# df_high_myopia_lax['AL_OS'].plot.hist(bins=100)
# plt.title("Distribution of high_myopia_lax-AL_OS")
# plt.grid()
# plt.show()
# plt.figure()
# df_high_myopia_lax['AL_OD'].plot.hist(bins=100)
# plt.title("Distribution of high_myopia_lax-AL_OD")
# plt.grid()
# plt.show()
# print(f"左眼眼轴长度大于26.5的人数:{df_high_myopia_lax[df_high_myopia_lax['AL_OS'] >= 26.5].shape[0]}")
# print(f"右眼眼轴长度大于26.5的人数:{df_high_myopia_lax[df_high_myopia_lax['AL_OD'] >= 26.5].shape[0]}")
# print(f"左眼眼轴长度大于24的人数:{df_high_myopia_lax[df_high_myopia_lax['AL_OS'] >= 24].shape[0]}")
# print(f"右眼眼轴长度大于24的人数:{df_high_myopia_lax[df_high_myopia_lax['AL_OD'] >= 24].shape[0]}")
# print(f"左眼眼轴长度大于23的人数:{df_high_myopia_lax[df_high_myopia_lax['AL_OS'] >= 23].shape[0]}")
# print(f"右眼眼轴长度大于23的人数:{df_high_myopia_lax[df_high_myopia_lax['AL_OD'] >= 23].shape[0]}")
# # 高度近视宽松标准2
# df_high_myopia_lax2 = df[df['SE-high_myopia_lax2'] == 1][['AL_OS', 'AL_OD']]
# df_high_myopia_lax2_len = df_high_myopia_lax2.shape[0]
# print(f'高度近视宽松标准2（一只眼小于-6D，另一只要小于-4D）人数：{df_high_myopia_lax2_len}')
# plt.figure()
# df_high_myopia_lax2['AL_OS'].plot.hist(bins=100)
# plt.title("Distribution of high_myopia_lax-AL_OS")
# plt.grid()
# plt.show()
# plt.figure()
# df_high_myopia_lax2['AL_OD'].plot.hist(bins=100)
# plt.title("Distribution of high_myopia_lax-AL_OD")
# plt.grid()
# plt.show()
# print(f"左眼眼轴长度大于26.5的人数:{df_high_myopia_lax2[df_high_myopia_lax2['AL_OS'] >= 26.5].shape[0]}")
# print(f"右眼眼轴长度大于26.5的人数:{df_high_myopia_lax2[df_high_myopia_lax2['AL_OD'] >= 26.5].shape[0]}")
# print(f"左眼眼轴长度大于24的人数:{df_high_myopia_lax2[df_high_myopia_lax2['AL_OS'] >= 24].shape[0]}")
# print(f"右眼眼轴长度大于24的人数:{df_high_myopia_lax2[df_high_myopia_lax2['AL_OD'] >= 24].shape[0]}")
# print(f"左眼眼轴长度大于23的人数:{df_high_myopia_lax2[df_high_myopia_lax2['AL_OS'] >= 23].shape[0]}")
# print(f"右眼眼轴长度大于23的人数:{df_high_myopia_lax2[df_high_myopia_lax2['AL_OD'] >= 23].shape[0]}")
# # 高度近视严格标准
# df_high_myopia_strict = df[df['SE-high_myopia_strict'] == 1][['AL_OS', 'AL_OD']]
# df_high_myopia_strict_len = df_high_myopia_strict.shape[0]
# print(f'高度近视严格标准（都小于-6D）人数：{df_high_myopia_strict_len}')
# plt.figure()
# df_high_myopia_strict['AL_OS'].plot.hist(bins=100)
# plt.title("Distribution of high_myopia_lax-AL_OS")
# plt.grid()
# plt.show()
# plt.figure()
# df_high_myopia_strict['AL_OD'].plot.hist(bins=100)
# plt.title("Distribution of high_myopia_lax-AL_OD")
# plt.grid()
# plt.show()
# print(f"左眼眼轴长度大于26.5的人数:{df_high_myopia_strict[df_high_myopia_strict['AL_OS'] >= 26.5].shape[0]}")
# print(f"右眼眼轴长度大于26.5的人数:{df_high_myopia_strict[df_high_myopia_strict['AL_OD'] >= 26.5].shape[0]}")
# print(f"左眼眼轴长度大于24的人数:{df_high_myopia_strict[df_high_myopia_strict['AL_OS'] >= 24].shape[0]}")
# print(f"右眼眼轴长度大于24的人数:{df_high_myopia_strict[df_high_myopia_strict['AL_OD'] >= 24].shape[0]}")
# print(f"左眼眼轴长度大于23的人数:{df_high_myopia_strict[df_high_myopia_strict['AL_OS'] >= 23].shape[0]}")
# print(f"右眼眼轴长度大于23的人数:{df_high_myopia_strict[df_high_myopia_strict['AL_OD'] >= 23].shape[0]}")
# 最终筛选结果
# df_high_myopia = df[(((df['SE_OS'] <= -6) & (df['SE_OD'] <= -4)) | ((df['SE_OD'] <= -6) & (df['SE_OS'] <= -4)))
#                     & (df['AL_OS'] >= 25) & (df['AL_OD'] >= 25)].copy()
# df_high_myopia = df[(((df['SE_OS'] <= -6) & (df['SE_OD'] <= -4)) | ((df['SE_OD'] <= -6) & (df['SE_OS'] <= -4)))
#                     & (df['AL_OS'] >= 26) & (df['AL_OD'] >= 26)].copy()
df_high_myopia = df[((df['SE_OS'] <= -6) & (df['AL_OS'] >= 26)) | ((df['SE_OD'] <= -6) & (df['AL_OD'] >= 26))].copy()
print(f'高度近视人数：{df_high_myopia.shape[0]}')
df_high_myopia.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort'
    rf'\all_students_qc_two_eyes_high_myopia_{datatime_now_str}.csv',
    index=False)
# 非高度近视
# df_not_high_myopia = df[(df['SE_OS'] >= -3) & (df['SE_OD'] >= -3) & (df['AL_OS'] <= 25) & (df['AL_OD'] <= 25)].copy()
# df_not_high_myopia = df[(df['SE_OS'] >= -3) & (df['SE_OD'] >= -3) & (df['AL_OS'] <= 24) & (df['AL_OD'] <= 24)].copy()
# df_not_high_myopia = df[(df['SE_OS'] >= -3) & (df['SE_OD'] >= -3) &
#                         (df['SE_OS'] <= -0.5) & (df['SE_OD'] <= -0.5) &
#                         (df['AL_OS'] <= 24) & (df['AL_OD'] <= 24)].copy()
df_not_high_myopia = df[(df['SE_OS'] >= -3) & (df['SE_OD'] >= -3) &
                        (df['SE_OS'] <= -0.5) & (df['SE_OD'] <= -0.5) &
                        (df['AL_OS'] <= 25) & (df['AL_OD'] <= 25)].copy()
print(f'非高度近视人数：{df_not_high_myopia.shape[0]}')
df_not_high_myopia.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort'
    rf'\all_students_qc_two_eyes_not_high_myopia_{datatime_now_str}.csv',
    index=False)
df_high_myopia['high_myopia'] = 1
df_not_high_myopia['high_myopia'] = 0
df_merge = pd.concat([df_high_myopia, df_not_high_myopia], axis=0, ignore_index=True)
df_merge.to_csv(
    rf'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort'
    rf'\all_students_qc_two_eyes_merge_{datatime_now_str}.csv',
    index=False)
