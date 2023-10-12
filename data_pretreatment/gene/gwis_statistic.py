# -*- encoding: utf-8 -*-
"""
@File Name      :   gwis_statistic.py   
@Create Time    :   2023/10/9 10:31
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

import json

import matplotlib.pyplot as plt
import numpy as np

with open(r"C:\Users\dikli\Downloads\group_wise_importance_score.json", 'r') as r:
    gwis = json.load(r)
# 加载数据
delta_losses = np.array(gwis['delta_losses'])
all_genes = np.array(gwis['all_genes'])
# 挑选delta_losses小于0的基因，这代表模型有效
selected_genes_rule = delta_losses < 0

selected_delta_losses = delta_losses[selected_genes_rule]
# 取前30%的数
selected_delta_losses = np.sort(selected_delta_losses)
selected_delta_losses_len = selected_delta_losses.shape[0]
percentile = np.percentile(selected_delta_losses, 30)
# 从数据中筛选出小于等于30%分位数的数
selected_delta_losses = selected_delta_losses[selected_delta_losses <= percentile]
# 设置间隔
step = 0.01
# 设置统计区间
bins = np.arange(np.min(selected_delta_losses), np.max(selected_delta_losses), step)  # 区间为0-100，每隔10为一个区间
# 统计数值出现的次数
hist, _ = np.histogram(selected_delta_losses, bins=bins)
# 绘制直方图
plt.bar(bins[:-1], hist, width=step)
# 设置图形标题和坐标轴标签
plt.title("Number Frequency")
plt.xlabel("Number")
plt.ylabel("Frequency")
# 显示图形
plt.show()

selected_genes = all_genes[selected_genes_rule]
selected_genes = selected_genes[selected_genes <= percentile]
print(len(selected_genes))