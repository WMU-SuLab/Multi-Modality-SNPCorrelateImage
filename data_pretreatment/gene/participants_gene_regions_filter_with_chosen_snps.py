# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_gene_regions_filter_with_chosen_snps.py
@Create Time    :   2023/9/14 11:27
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

import os

import click
import pandas as pd


@click.command()
@click.argument('chosen_snps_file_path', type=click.Path(exists=True))
@click.argument('input_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
def main(chosen_snps_file_path, input_dir_path, output_dir_path):
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    with open(chosen_snps_file_path, 'r') as f:
        columns = f.read().split(',')
        snp_ids = columns[1:]
        snp_ids = [f'chr{i}' for i in snp_ids]
    for file_name in os.listdir(input_dir_path):
        if file_name.endswith('.csv') and file_name != 'columns.csv':
            df = pd.read_csv(os.path.join(input_dir_path, file_name))
            df = df[df['snp_id'].isin(snp_ids)]
            df.to_csv(os.path.join(output_dir_path, file_name), index=None)


if __name__ == '__main__':
    """
    participant_id.csv过滤成gene_regions.csv之后，可能要过滤一部分根据条件筛选出来的snp
    """
    main()
