# -*- encoding: utf-8 -*-
"""
@File Name      :   fake_label.py
@Create Time    :   2023/4/12 16:45
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

import random

import pandas as pd

df = pd.read_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\ftd_myopia_left.csv')
df_len = df.shape[0]
df['是否高度近视-SE'] = [random.randint(0, 1) for i in range(df_len)]
df.to_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\ftd_myopia_left_fake.csv',
          index=False)
