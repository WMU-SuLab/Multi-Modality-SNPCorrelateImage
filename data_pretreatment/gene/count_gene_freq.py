# -*- encoding: utf-8 -*-
"""
@File Name      :   count_gene_freq.py
@Create Time    :   2023/5/9 9:59
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

import json
import os

import click


@click.command()
@click.argument('gene_data_dir_path', type=click.Path(exists=True))
@click.option('--output_dir_path', type=click.Path(exists=True), default=None)
@click.option('--output_file_name', type=str, default='freq.json')
def main(gene_data_dir_path: str, output_dir_path: str, output_file_name: str):
    gene_data_columns_len = 0
    gene_data_file_paths = [
        os.path.join(gene_data_dir_path, gene_data_file_name)
        for gene_data_file_name in os.listdir(gene_data_dir_path) if gene_data_file_name.endswith('.csv')]
    for gene_data_file_path in gene_data_file_paths:
        gene_data_file = open(gene_data_file_path, 'r')
        gene_data = gene_data_file.read().strip().split(',')[1:]
        gene_data_file.close()
        if gene_data_columns_len == 0:
            gene_data_columns_len = len(gene_data)
            gene_freq = [{'0': 0, '1': 0, '2': 0} for i in range(gene_data_columns_len)]
        for index, gene in enumerate(gene_data):
            gene_freq[index][gene] += 1
    for index in range(gene_data_columns_len):
        gene_freq[index]['0'] /= gene_data_columns_len
        gene_freq[index]['1'] /= gene_data_columns_len
        gene_freq[index]['2'] /= gene_data_columns_len
    if output_dir_path is None:
        output_dir_path = os.path.dirname(gene_data_dir_path)
    with open(os.path.join(output_dir_path, output_file_name), 'w') as output_file:
        # ensure_ascii=False才能输入中文，否则是Unicode字符
        # indent=2 JSON数据的缩进，美观
        json.dump(gene_freq, output_file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
