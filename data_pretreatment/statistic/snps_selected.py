# -*- encoding: utf-8 -*-
"""
@File Name      :   snps.py   
@Create Time    :   2023/10/22 15:13
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

import os
from collections import defaultdict

base_dir = os.path.join('work_dirs', 'data', 'gene')

P = [0.001, 0.005, 0.01, 0.05]
OR = [1.0, 1.5, 2.0]
percentile = [5, 10, 20, 30]
P_OR_snps = defaultdict(list)
for p in P:
    for or_val in OR:
        with open(os.path.join(f'students_snps_P_{p}_OR_{or_val}_S8114_final_fastgwa_SE1', 'columns.csv'), 'r') as f:
            P_OR_snps[f'P_{p}_OR_{or_val}'] = f.read().split(',')[1:]

P_OR_set = set().union(*[set(val) for val in P_OR_snps.values()])
print(len(P_OR_set))

P_Percentile_snps = defaultdict(list)
for p in P:
    for per in percentile:
        with open(os.path.join(f'students_snps_all_frequency_{p}', f'selected_genes_percentile_{per}', 'columns.csv'),
                  'r') as f:
            P_Percentile_snps[f'P_{p}_percentile_{per}'] = f.read().split(',')[1:]
P_Percentile_set = set().union(*[set(val) for val in P_Percentile_snps.values()])
print(len(P_Percentile_set))

all_set = P_OR_set.union(P_Percentile_set)
print(len(all_set))
