# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_gene_regions_filter_snps_with_selected_gene.py
@Create Time    :   2023/9/8 20:42
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
import tqdm
import click
import numpy as np
import pandas as pd


@click.command()
@click.argument('input_selected_gene_file_path', type=click.Path(exists=True))
@click.argument('input_data_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
@click.option('-t', '--threshold', type=float, default=0)
@click.option('-g', '--generate', is_flag=True, help='generate snps.csv')
def main(input_selected_gene_file_path, input_data_dir_path, output_dir_path, threshold, generate):
    # 使用 group-wise importance score 得到的基因区域提取数据
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)

    if input_selected_gene_file_path.endswith('.csv') or input_selected_gene_file_path.endswith('.txt'):
        with open(input_selected_gene_file_path, 'r') as r:
            selected_genes = r.read().split(',')
    elif input_selected_gene_file_path.endswith('.json'):
        with open(input_selected_gene_file_path, 'r') as r:
            gwis = json.load(r)
        delta_losses = np.array(gwis['delta_losses'])
        all_genes = np.array(gwis['all_genes'])
        print(f'all_genes: {len(all_genes)}')
        if not threshold:
            raise ValueError('threshold must be set')
        effective_genes_rule = delta_losses < 0
        effective_genes = all_genes[effective_genes_rule]
        print(f'effective_genes: {len(effective_genes)}')
        effective_delta_losses = delta_losses[effective_genes_rule]
        percentile = np.percentile(effective_delta_losses, threshold)
        selected_genes = effective_genes[effective_delta_losses <= percentile]
    else:
        raise ValueError('input_selected_gene_file_path must be .csv, .txt or .json')
    print(f'selected_genes: {len(selected_genes)}')
    # 根据筛选出的基因构建snps文件
    if generate:
        # participant_file_names = os.listdir(input_data_dir_path)
        # with open(os.path.join(output_dir_path, f'columns.csv'), 'w') as f:
        #     df = pd.read_csv(os.path.join(input_data_dir_path, participant_file_names[0]))
        #     df = df[df['gene'].isin(selected_genes)]
        #     snp_ids = df['snp_id'].tolist()
        #     print(f'snp_ids: {len(snp_ids)}')
        #     f.write('participant_id,' + ','.join(snp_ids))
        # for file_name in participant_file_names:
        #     participant_id = file_name.split('.')[0]
        #     if not file_name.endswith('.csv'):
        #         continue
        #     df = pd.read_csv(os.path.join(input_data_dir_path, file_name), dtype=str)
        #     df = df[df['gene'].isin(selected_genes)]
        #     with open(os.path.join(output_dir_path, f'{participant_id}.csv'), 'w') as f:
        #         f.write(f'{participant_id},' + ','.join(df['val'].tolist()))

        selected_genes_file_names = [f"{selected_gene}.csv" for selected_gene in selected_genes]
        df = pd.read_csv(os.path.join(input_data_dir_path, selected_genes_file_names[0]), dtype=str)
        participant_ids = df['participant_id'].drop_duplicates().tolist()
        participant_files = {participant_id: open(os.path.join(output_dir_path, f'{participant_id}.csv'), 'w')
                             for participant_id in participant_ids}
        for participant_id, participant_file in participant_files.items():
            participant_file.write(f'{participant_id},')
        with open(os.path.join(output_dir_path, f'columns.csv'), 'w') as column_file:
            column_file.write('participant_id,')
            for selected_genes_file_name in tqdm.tqdm(selected_genes_file_names):
                gene_df = pd.read_csv(os.path.join(input_data_dir_path, selected_genes_file_name), dtype=str)
                snp_ids = gene_df['snp_id'].drop_duplicates().tolist()
                column_file.write(','.join(snp_ids))
                for participant_id, participant_group in gene_df.groupby('participant_id'):
                    participant_file = participant_files[participant_id]
                    participant_file.write(','.join(participant_group['val'].astype(str).tolist()))
        for participant_file in participant_files.values():
            participant_file.close()


if __name__ == '__main__':
    """
    通过 gwis 方法筛选出来基因，将gene_regions文件夹中的基因文件重新组织为participant_snps的形式去训练
    python data_pretreatment/gene/participants_gene_regions_filter_snps_with_selected_gene.py \
    work_dirs/data/gene/students_snps_all_frequency_0.005/gene_regions/group_wise_importance_score.json \
    work_dirs/data/gene/students_snps_all_frequency_0.005/gene_regions \
    work_dirs/data/gene/students_snps_all_frequency_0.005/selected_gene_threshold_2 --threshold 2
    
    
    """
    main()
