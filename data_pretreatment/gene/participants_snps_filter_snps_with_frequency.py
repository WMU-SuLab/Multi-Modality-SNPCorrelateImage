# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_snps_filter_snps_with_frequency.py
@Create Time    :   2023/9/14 11:21
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

import click
import numpy as np


@click.command()
@click.argument('input_data_dir_path', type=click.Path(exists=True))
@click.argument('threshold', type=float)
@click.argument('output_dir_path', type=click.Path())
@click.argument('frequency', type=float)
@click.option('--write_statistic_file', is_flag=True, help='write statistic.json')
@click.option('--method', default='none', type=click.Choice(['none', 'all', 'frequency', 'file']), help='methods')
def main(input_data_dir_path, threshold: float, output_dir_path, frequency: float, write_statistic_file, method):
    if output_dir_path and (not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path)):
        os.mkdir(output_dir_path)
    if frequency < 0 or frequency > 1:
        raise ValueError('frequency must be between 0 and 1')

    if write_statistic_file:
        with open(os.path.join(input_data_dir_path, 'columns.csv'), 'r') as f:
            columns = f.read().split(',')
        snp_ids = np.array(columns[1:], dtype=str)

        snps_divide_count = {
            0: np.zeros(len(snp_ids), dtype=int),
            1: np.zeros(len(snp_ids), dtype=int),
            2: np.zeros(len(snp_ids), dtype=int)
        }
        participants_count = 0
        for file_name in os.listdir(input_data_dir_path):
            if file_name.endswith('.csv') and file_name != 'columns.csv':
                participants_count += 1
                file_path = os.path.join(input_data_dir_path, file_name)
                with open(file_path, 'r') as f:
                    snp_columns = f.read().split(',')
                    snps = np.array(snp_columns[1:], dtype=int)
                    for key in snps_divide_count.keys():
                        new_snps = np.where(snps == key, 1, 0)
                        snps_divide_count[key] += new_snps

        with open(os.path.join(input_data_dir_path, 'statistic.json'), 'w') as f:
            print('writing statistic.json')
            json.dump({
                'participants_count': participants_count,
                'snp_ids': snp_ids.tolist(),
                'snps_divide_count': {key: val.tolist() for key, val in snps_divide_count.items()},
                'snps_divide_frequency': {key: (val / participants_count).tolist() for key, val in
                                          snps_divide_count.items()},
            }, f)
    else:
        with open(os.path.join(input_data_dir_path, 'columns.csv'), 'r') as f:
            columns = f.read().split(',')
        with open(os.path.join(input_data_dir_path, 'statistic.json'), 'r') as f:
            statistic = json.load(f)
        participants_count = statistic['participants_count']
        snp_ids = np.array(statistic['snp_ids'], dtype=str)
        snps_divide_count = {int(key): np.array(val, dtype=int) for key, val in statistic['snps_divide_count'].items()}
    snps_threshold_rule = snps_divide_count[2] > threshold
    snps_divide_count[0][snps_threshold_rule], snps_divide_count[2][snps_threshold_rule] = snps_divide_count[2][
        snps_threshold_rule], snps_divide_count[0][snps_threshold_rule]
    snps_frequency = (snps_divide_count[1] + snps_divide_count[2]) / participants_count
    with open(os.path.join(input_data_dir_path, f'threshold_{threshold}_frequency.json'), 'w') as f:
        print(f'writing threshold_{threshold}_frequency.json')
        json.dump({
            'threshold_snps': snp_ids[snps_threshold_rule].tolist(),
            'snps_frequency': {snp_id: snp_freq for snp_id, snp_freq in zip(snp_ids.tolist(), snps_frequency.tolist())}
        }, f)
    # 其实有了前面的0 2对换，是不需要 snps_frequency < 1 - frequency 的
    filter_snps_rule = np.logical_and(snps_frequency > frequency, snps_frequency < 1 - frequency)
    filtered_snp_ids = snp_ids[filter_snps_rule]
    if method == 'all' or method == 'frequency':
        with open(os.path.join(output_dir_path, f'filter_snps_frequency_{frequency}.json'), 'w') as f:
            print(f'writing filter_snps_frequency_{frequency}.json')
            json.dump({snp_id: snp_num for snp_id, snp_num in
                       zip(filtered_snp_ids.tolist(), snps_frequency[filter_snps_rule].tolist())}, f)
    if method == 'all' or method == 'file':
        print('writing participants')
        with open(os.path.join(output_dir_path, f'columns.csv'), 'w') as f:
            f.write(f'{columns[0]},' + ','.join(filtered_snp_ids))
        for file_name in os.listdir(input_data_dir_path):
            if file_name.endswith('.csv') and file_name != 'columns.csv':
                file_path = os.path.join(input_data_dir_path, file_name)
                output_file_path = os.path.join(output_dir_path, file_name)
                with open(file_path, 'r') as f:
                    snp_columns = f.read().split(',')
                    snps = np.array(snp_columns[1:], dtype=str)
                    snps[snps_threshold_rule] = list(map(lambda x: 2 if x == 0 else 0, snps[snps_threshold_rule]))
                    snps = snps[filter_snps_rule]
                with open(output_file_path, 'w') as f:
                    f.write(f'{snp_columns[0]},' + ','.join(snps))


if __name__ == '__main__':
    """
    从所有snps中根据条件筛选snps，是participants_frequency_snps_to_gene_regions.py的前置条件
    """
    main()
