# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_gene_regions_regenerate_info.py   
@Create Time    :   2023/10/6 15:16
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
import os

import pandas as pd
import tqdm

dir_path = 'work_dirs/data/gene/students_snps_all_frequency_0.001/gene_regions/'
gene_names = []
gene_regions_len = {}
file_names = [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.csv')]
for file_name in tqdm.tqdm(file_names):
    gene_name = file_name.split('.')[0]
    gene_names.append(gene_name)
    df = pd.read_csv(os.path.join(dir_path, file_name), dtype=str)
    groups = df.groupby('participant_id')
    gene_regions_len[gene_name] = list(groups)[0][1].shape[0]

with open(os.path.join(dir_path, 'gene_regions_info.json'), 'w') as f:
    json.dump({
        'gene_names': gene_names,
        'gene_region_snps_len': gene_regions_len,
    }, f)
