# -*- encoding: utf-8 -*-
"""
@File Name      :   barcodes.py
@Create Time    :   2023/4/25 19:48
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

import os

import pandas as pd

df = pd.read_csv(r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_barcodes.csv')
print(df.shape)
overlap_df = pd.read_excel(r'D:\BaiduSyncdisk\Data\SuLabCohort\label\重叠样本.xlsx', sheet_name='Sheet1')
overlap_dict = {row['样本编号']: row['上次编号'] for index, row in overlap_df.iterrows()}
for index, row in df.iterrows():
    if last_barcode := overlap_dict.get(row['条形码'], None):
        df.loc[index, '上次编号'] = last_barcode
df.to_csv(
    r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_barcodes_with_overlap.csv',
    index=False)
gene_files = os.listdir(r'D:\BaiduSyncdisk\Data\SuLabCohort\gene\students_snps_MLMALOCO_0.05')
gene_students_barcodes = [i.split('.')[0] for i in gene_files]
print(len(gene_students_barcodes))
students_barcodes = df['条形码'].dropna().tolist()
students_barcodes_with_snp = set(students_barcodes) & set(gene_students_barcodes)
print(len(students_barcodes_with_snp))
students_barcodes_with_snp_df = df[df['条形码'].isin(students_barcodes_with_snp)]
print(students_barcodes_with_snp_df.shape)
students_barcodes_with_snp_df.to_csv(
    r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_barcodes_with_snp.csv',
    index=False)
students_barcodes_without_snp = list(set(students_barcodes) - set(students_barcodes_with_snp))
print(len(students_barcodes_without_snp))
students_barcodes_without_snp_df = df[df['条形码'].isin(students_barcodes_without_snp)]
print(students_barcodes_without_snp_df.shape)
students_barcodes_without_snp_df.to_csv(
    r'D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\label\SuLabCohort\all_students_barcodes_without_snp.csv',
    index=False)
