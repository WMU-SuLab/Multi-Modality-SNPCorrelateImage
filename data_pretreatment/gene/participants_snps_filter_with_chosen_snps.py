# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_snps_filter_with_chosen_snps.py   
@Create Time    :   2024/1/7 20:08
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
from operator import itemgetter

import click


@click.command()
@click.argument('chosen_snps_file_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_snps_dir_path', type=click.Path(exists=True, dir_okay=True))
@click.option('-o', '--output_dir_path', type=click.Path(dir_okay=True), default='')
def main(chosen_snps_file_path, input_snps_dir_path, output_dir_path):
    with open(chosen_snps_file_path) as f:
        chosen_snp_ids = f.read().strip().split(',')
        print(f'chosen_snp_ids:{len(chosen_snp_ids)}')
    output_dir_path = os.path.dirname(chosen_snps_file_path) or output_dir_path
    with open(os.path.join(input_snps_dir_path, 'columns.csv'), 'r') as f:
        all_snp_ids = f.read().strip().split(',')
    chosen_snps_index = [all_snp_ids.index(i) for i in chosen_snp_ids]
    print(f"chosen_snps_index:{len(chosen_snps_index)}")
    if not chosen_snps_index:
        print('no snp chosen')
        return
    for file_name in os.listdir(input_snps_dir_path):
        if file_name.endswith('.csv'):
            with open(os.path.join(input_snps_dir_path, file_name), 'r') as f:
                snp_vals = f.read().strip().split(',')
            chosen_snp_vals = list(itemgetter(*chosen_snps_index)(snp_vals))
            output_file_path = os.path.join(output_dir_path, file_name)
            with open(output_file_path, 'w') as f:
                f.write(','.join([snp_vals[0]] + chosen_snp_vals))


if __name__ == '__main__':
    main()
