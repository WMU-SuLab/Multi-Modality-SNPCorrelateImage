# -*- encoding: utf-8 -*-
"""
@File Name      :   eye_all.py
@Create Time    :   2023/10/18 16:18
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@other information
"""
__auth__ = 'diklios'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import matplotlib
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# matplotlib.rcParams['axes.unicode_minus']=False
# sns.set_context("talk")


rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(context='notebook', style='ticks', rc=rc)
df = pd.read_csv(r"D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_qc.csv")
df = df[['SE_OS', 'SE_OD', 'AL_OS', 'AL_OD']]

# 使用 Seaborn 设置样式
# sns.set_style("whitegrid", {'axes.grid': False})

# 绘制SE直方图
# sns.displot(df, x='SE_OS', binwidth=0.5, shrink=.8, discrete=True, )
# plt.show()
# sns.displot(df, x='SE_OD', binwidth=0.5, shrink=.8, discrete=True, )
# plt.show()
# 合并
df_SE_OS = df[['SE_OS']]
df_SE_OS = df_SE_OS.rename(columns={'SE_OS': '等效球镜'})
df_SE_OS['左右眼'] = '左眼'
df_SE_OD = df[['SE_OD']]
df_SE_OD = df_SE_OD.rename(columns={'SE_OD': '等效球镜'})
df_SE_OD['左右眼'] = '右眼'
df_SE = pd.concat([df_SE_OS, df_SE_OD], axis=0, ignore_index=True)
sns.displot(df_SE, x='等效球镜', hue="左右眼", binwidth=0.5, shrink=.8, discrete=True, )

# 绘制AL直方图
# sns.displot(df, x='AL_OS', binwidth=0.5, shrink=.8, discrete=True, )
# plt.show()
# sns.displot(df, x='AL_OD', binwidth=0.5, shrink=.8, discrete=True, )
# plt.show()
# 合并
df_AL_OS = df[['AL_OS']]
df_AL_OS = df_AL_OS.rename(columns={'AL_OS': '眼轴长'})
df_AL_OS['左右眼'] = '左眼'
df_AL_OD = df[['AL_OD']]
df_AL_OD = df_AL_OD.rename(columns={'AL_OD': '眼轴长'})
df_AL_OD['左右眼'] = '右眼'
df_AL = pd.concat([df_AL_OS, df_AL_OD], axis=0, ignore_index=True)
sns.displot(df_AL, x='眼轴长', hue="左右眼", binwidth=0.5, shrink=.8, discrete=True, )
plt.show()
