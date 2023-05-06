# -*- encoding: utf-8 -*-
"""
@File Name      :   myopia.py
@Create Time    :   2023/3/2 11:27
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

import pandas as pd

df = pd.read_csv(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\左眼特征.csv', encoding='gb18030', dtype={'学籍号': str})
df['是否高度近视-SE'] = df.apply(lambda x: 1 if x['SE(左眼)'] <= -6 else 0, axis=1)
df['是否高度近视-AL'] = df.apply(lambda x: 1 if x['左眼眼轴长度(AL)'] <= -26.5 else 0, axis=1)
df.to_csv(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\myopia_left.csv', encoding='utf-8', index=False)

df2 = pd.read_csv(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\右眼特征.csv', encoding='gb18030', dtype={'学籍号': str})
df2['是否高度近视-SE'] = df2.apply(lambda x: 1 if x['SE(右眼)'] <= -6 else 0, axis=1)
df2['是否高度近视-AL'] = df2.apply(lambda x: 1 if x['右眼眼轴长度(AL)'] <= -26.5 else 0, axis=1)
df2.to_csv(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\myopia_right.csv', encoding='utf-8', index=False)
