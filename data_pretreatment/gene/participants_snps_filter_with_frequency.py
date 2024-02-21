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
import tqdm


@click.command()
@click.argument('input_data_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
@click.argument('frequency', type=float)
@click.option('--write_statistic_file', is_flag=True, help='write statistic.json')
@click.option('--method', type=click.Choice(['common', 'rare', 'other']), default='common', help='SNP choose method')
# @click.option('--threshold', type=float, default=0, help='0-2 对换时候的threshold')
def main(input_data_dir_path, output_dir_path, frequency: float, write_statistic_file, method: str,
         # threshold: float,
         ):
    if output_dir_path and (not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path)):
        os.mkdir(output_dir_path)
    if frequency < 0 or frequency > 1:
        raise ValueError('frequency must be between 0 and 1')

    if write_statistic_file:
        with open(os.path.join(input_data_dir_path, 'columns.csv'), 'r') as f:
            columns = f.read().split(',')
        snp_ids = np.array(columns[1:], dtype=str)
        snps_divide_count = {
            -1: np.zeros(len(snp_ids), dtype=int),
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
    # 0-2 对换
    # if 0 < threshold < 1:
    #     snps_threshold_rule = (snps_divide_count[2] / participants_count) > threshold
    #     snps_divide_count[0][snps_threshold_rule], snps_divide_count[2][snps_threshold_rule] = snps_divide_count[2][
    #         snps_threshold_rule], snps_divide_count[0][snps_threshold_rule]
    #     snps_frequency = (snps_divide_count[1] + snps_divide_count[2]) / participants_count
    #     with open(os.path.join(input_data_dir_path, f'02_threshold_{threshold}_frequency.json'), 'w') as f:
    #         print(f'writing threshold_{threshold}_frequency.json')
    #         json.dump({
    #             'threshold_snps': snp_ids[snps_threshold_rule].tolist(),
    #             'snps_frequency': {snp_id: snp_freq for snp_id, snp_freq in zip(snp_ids.tolist(), snps_frequency.tolist())}
    #         }, f)
    # else:
    snps_frequency_12 = (snps_divide_count[1] / 2 + snps_divide_count[2]) / participants_count
    snps_frequency_01 = (snps_divide_count[0] + snps_divide_count[1] / 2) / participants_count
    if method == 'common':
        filter_snps_rule = np.logical_and(snps_frequency_12 >= frequency, snps_frequency_01 <= (1 - frequency))
    elif method == 'rare':
        filter_snps_rule = np.logical_or(snps_frequency_12 < frequency, snps_frequency_01 > (1 - frequency))
    elif method == 'other':
        # other是给一些自定义的情况设计的，下面这部分代码可以自己修改
        filter_snps_rule = np.logical_or(np.logical_and(snps_frequency_12 >= 0.0001, snps_frequency_12 <= 0.001),
                                         np.logical_and(snps_frequency_01 > 0.999, snps_frequency_01 < 0.9999))
    else:
        raise ValueError('method invalid')
    filtered_snp_ids = snp_ids[filter_snps_rule]
    print('writing participants')
    with open(os.path.join(output_dir_path, f'columns.csv'), 'w') as f:
        f.write(f'{columns[0]},' + ','.join(filtered_snp_ids))
    for file_name in tqdm.tqdm(os.listdir(input_data_dir_path)):
        if file_name.endswith('.csv') and file_name != 'columns.csv':
            file_path = os.path.join(input_data_dir_path, file_name)
            output_file_path = os.path.join(output_dir_path, file_name)
            with open(file_path, 'r') as f:
                snp_columns = f.read().split(',')
                snps = np.array(snp_columns[1:], dtype=str)
                # 0-2 对换
                # if 0 < threshold < 1:
                #     snps_0_rule = np.logical_and(snps_threshold_rule, snps == 0)
                #     snps_2_rule = np.logical_and(snps_threshold_rule, snps == 2)
                #     snps[snps_0_rule] = 2
                #     snps[snps_2_rule] = 0
                snps = snps[filter_snps_rule]
            with open(output_file_path, 'w') as f:
                f.write(f'{snp_columns[0]},' + ','.join(snps))


if __name__ == '__main__':
    """
    从所有snps中根据条件筛选snps，是participants_frequency_snps_to_gene_regions.py的前置条件
    示例（自定义）：python data_pretreatment/gene/participants_snps_filter_snps_with_frequency.py work_dirs/data/gene/students_snps_all \
    work_dirs/data/gene/students_snps_all_frequency_0.0001_0.001 0 --set_method other --method all
    示例（rare）：python data_pretreatment/gene/participants_snps_filter_with_frequency.py \
    work_dirs/data/gene/students_snps_all_missing/ work_dirs/data/gene/students_snps_all_missing_frequency_0.05_common \
    0.05 --method common
    示例（rare）：python data_pretreatment/gene/participants_snps_filter_with_frequency.py \
    work_dirs/data/gene/students_snps_all_missing/ work_dirs/data/gene/students_snps_all_missing_frequency_0.05_rare \
    0.05 --method rare
    """
    main()
