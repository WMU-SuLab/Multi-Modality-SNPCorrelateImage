# -*- encoding: utf-8 -*-
"""
@File Name      :   eye_filter_two.py   
@Create Time    :   2023/10/19 11:39
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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc.csv')
df = df[['SE_OS', 'SE_OD', 'AL_OS', 'AL_OD']]

# old
# df_high_myopia = df[(((df['SE_OS'] <= -6) & (df['SE_OD'] <= -4)) | ((df['SE_OD'] <= -6) & (df['SE_OS'] <= -4)))
#                     & (df['AL_OS'] >= 25) & (df['AL_OD'] >= 25)].copy()
# df_not_high_myopia = df[(df['SE_OS'] >= -3) & (df['SE_OD'] >= -3) & (df['AL_OS'] <= 25) & (df['AL_OD'] <= 25)].copy()

# new
df_high_myopia = df[((df['SE_OS'] <= -6) & (df['AL_OS'] >= 26)) | ((df['SE_OD'] <= -6) & (df['AL_OD'] >= 26))].copy()
df_not_high_myopia = df[(df['SE_OS'] >= -3) & (df['SE_OD'] >= -3) &
                        (df['SE_OS'] <= -0.5) & (df['SE_OD'] <= -0.5) &
                        (df['AL_OS'] <= 25) & (df['AL_OD'] <= 25)].copy()

high_myopia_df = pd.DataFrame({
    'SE': df_high_myopia['SE_OS'].tolist() + df_high_myopia['SE_OD'].tolist(),
    'AL': df_high_myopia['AL_OS'].tolist() + df_high_myopia['AL_OD'].tolist(),
    'SE_species': 'high_myopia',
    'AL_species': 'high_myopia'
})
not_high_myopia_df = pd.DataFrame({
    'SE': df_not_high_myopia['SE_OS'].tolist() + df_not_high_myopia['SE_OD'].tolist(),
    'AL': df_not_high_myopia['AL_OS'].tolist() + df_not_high_myopia['AL_OD'].tolist(),
    'SE_species': 'not_high_myopia',
    'AL_species': 'not_high_myopia'
})

all_df = pd.concat([high_myopia_df, not_high_myopia_df], axis=0, ignore_index=True)

# 使用 Seaborn 设置样式
sns.set_style("whitegrid", {'axes.grid': False})

# 绘制SE直方图
sns.displot(all_df, x='SE', hue='SE_species', binwidth=0.5, shrink=.8, discrete=True, )
plt.show()
sns.catplot(data=all_df, x="SE_species", y="SE", kind="violin")
plt.show()
SE_species_counts = all_df['SE_species'].value_counts()
print(SE_species_counts)
plt.pie(SE_species_counts, labels=SE_species_counts.index, startangle=90, pctdistance=0.7, labeldistance=1.2,
        # colors=['deepskyblue', 'coral'],
        colors=['#6baed6', "#fd8d3c"],
        counterclock=False, wedgeprops={"linewidth": 1, "edgecolor": "white"}, autopct='%3.1f%%',
        )
plt.show()
# 绘制AL直方图
sns.displot(all_df, x='AL', hue='AL_species', binwidth=0.5, shrink=.8, discrete=True, )
plt.show()
sns.catplot(data=all_df, x="AL_species", y="AL", kind="violin")
plt.show()
