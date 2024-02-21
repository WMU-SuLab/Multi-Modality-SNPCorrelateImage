# -*- encoding: utf-8 -*-
"""
@File Name      :   snps_frequency.py   
@Create Time    :   2023/10/22 15:34
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

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})

with open(os.path.join('../', '../', 'work_dirs', 'data', 'gene', 'students_snps_all', 'statistic.json'), 'r') as f:
    statistic = json.load(f)

participants_count = statistic['participants_count']
snp_ids = np.array(statistic['snp_ids'], dtype=str)
snps_divide_count = {int(key): np.array(val, dtype=int) for key, val in statistic['snps_divide_count'].items()}

for key, val in snps_divide_count.items():
    val = val / participants_count
    sns.displot(pd.DataFrame({f'{key}-frequency': val}), x=f'{key}-frequency', binwidth=0.01)
snps_frequency = (snps_divide_count[1] + snps_divide_count[2]) / participants_count
snps_frequency = snps_frequency[snps_frequency >= 0.001]
sns.displot(pd.DataFrame({'1+2frequency': snps_frequency}), x='1+2frequency', binwidth=0.01)
plt.show()

threshold = 0.9
snps_threshold_rule = snps_divide_count[2] / participants_count > threshold
snps_divide_count[0][snps_threshold_rule], snps_divide_count[2][snps_threshold_rule] = snps_divide_count[2][
    snps_threshold_rule], snps_divide_count[0][snps_threshold_rule]
for key, val in snps_divide_count.items():
    val = val / participants_count
    sns.displot(pd.DataFrame({f'exchange 0-2 {key}-frequency': val}), x=f'exchange 0-2 {key}-frequency', binwidth=0.01)
snps_frequency = (snps_divide_count[1] + snps_divide_count[2]) / participants_count
snps_frequency = snps_frequency[snps_frequency >= 0.001]
sns.displot(pd.DataFrame({'exchange 0-2 1+2frequency': snps_frequency}), x='exchange 0-2 1+2frequency', binwidth=0.01)
plt.show()
