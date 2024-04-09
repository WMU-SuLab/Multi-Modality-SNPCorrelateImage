# -*- encoding: utf-8 -*-
"""
@File Name      :   gwis_statistic.py   
@Create Time    :   2023/11/7 15:23
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 使用 Seaborn 设置样式
sns.set_style("whitegrid", {'axes.grid': False})

percentages = [5, 10, 20, 30]
with open(r"C:\Users\dikli\Downloads\group_wise_importance_score.json", 'r') as r:
    gwis = json.load(r)
# 加载数据
delta_losses = np.array(gwis['delta_losses'])
all_genes = np.array(gwis['all_genes'])
# 挑选delta_losses小于0的基因，这代表模型有效
selected_genes_rule = delta_losses < 0
selected_delta_losses = delta_losses[selected_genes_rule]
selected_genes = all_genes[selected_genes_rule]
sns.displot(pd.DataFrame({f'distribution': selected_delta_losses}),
            x=f'distribution', binwidth=0.01)
# 显示图形
plt.show()
for percentage in percentages:
    print(f'percentage={percentage}')
    new_selected_delta_losses = selected_delta_losses.copy()
    selected_delta_losses_len = new_selected_delta_losses.shape[0]
    percentile = np.percentile(new_selected_delta_losses, percentage)
    # 从数据中筛选出小于等于30%分位数的数
    percentile_rule = new_selected_delta_losses <= percentile
    new_selected_delta_losses = new_selected_delta_losses[percentile_rule]

    sns.displot(pd.DataFrame({f'{percentage}distribution': new_selected_delta_losses}),
                x=f'{percentage}distribution', binwidth=0.01)
    # 显示图形
    plt.show()

    print(selected_genes[percentile_rule][new_selected_delta_losses < -3])

