# -*- encoding: utf-8 -*-
"""
@File Name      :   distribute.py
@Create Time    :   2022/10/31 15:32
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

from base import all_participants_file_path, AMD_file_path, no_AMD_file_path, DR_file_path, no_DR_file_path, GLC_file_path, \
    no_GLC_file_path, RD_file_path, no_RD_file_path, no_disease_file_path

# 处理疾病的数据
# 筛选数据
all_participants_df = pd.read_csv(all_participants_file_path)

AMD_df = all_participants_df[all_participants_df['AMD'] == 1]
AMD_df.to_csv(AMD_file_path, index=False)
no_AMD_df = all_participants_df[all_participants_df['AMD'] == 0]
no_AMD_df.to_csv(no_AMD_file_path, index=False)
DR_df = all_participants_df[all_participants_df['DR'] == 1]
DR_df.to_csv(DR_file_path, index=False)
no_DR_df = all_participants_df[all_participants_df['DR'] == 0]
no_DR_df.to_csv(no_DR_file_path, index=False)
GLC_df = all_participants_df[all_participants_df['GLC'] == 1]
GLC_df.to_csv(GLC_file_path, index=False)
no_GLC_df = all_participants_df[all_participants_df['GLC'] == 0]
no_GLC_df.to_csv(no_GLC_file_path, index=False)
RD_df = all_participants_df[all_participants_df['RD'] == 1]
RD_df.to_csv(RD_file_path, index=False)
no_RD_df = all_participants_df[all_participants_df['RD'] == 0]
no_RD_df.to_csv(no_RD_file_path, index=False)
no_disease_df = all_participants_df[
    (all_participants_df['AMD'] == 0) &
    (all_participants_df['DR'] == 0) &
    (all_participants_df['GLC'] == 0) &
    (all_participants_df['RD'] == 0)
    ]
no_disease_df.to_csv(no_disease_file_path, index=False)
