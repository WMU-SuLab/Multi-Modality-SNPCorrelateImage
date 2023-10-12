# -*- encoding: utf-8 -*-
"""
@File Name      :   vcf_speed_up_full.py
@Create Time    :   2022/12/8 16:55
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

import codecs
import gzip
import os
from operator import itemgetter
from time import time

import click
import pandas as pd

from base import snp_info_columns, regularize_rules


@click.command()
@click.argument('output_dir_path', type=click.Path())
@click.option('--input_file_path', type=str, default=None)
@click.option('--input_dir_path', type=str, default=None)
@click.option('-f', '--filtered_participants_file_path', help='限制的参与者的文件路径')
@click.option('-s', '--skip_rows', default=0, help='跳过的行数')
@click.option('--sep', default='\t', help='分隔符')
@click.option('-r', '--regularize', is_flag=True, help='是否规范化')
@click.option('--samples_ratio', default=0.001, type=float, help='样本比例')
def main(output_dir_path: str, input_file_path: str, input_dir_path: str,
         filtered_participants_file_path: str, skip_rows: int, sep: str,
         regularize: bool, samples_ratio: float):
    if input_file_path:
        input_file_path = os.path.abspath(input_file_path)
        input_file_paths = [input_file_path]
    elif input_dir_path:
        input_dir_path = os.path.abspath(input_dir_path)
        input_file_names = [file_name for file_name in os.listdir(input_dir_path) if file_name.endswith('.vcf.gz')]
        input_file_paths = [os.path.join(input_dir_path, file_name) for file_name in input_file_names]
    else:
        raise ValueError('input_file_path 和 input_dir_path 不能同时为空')
    print(input_file_paths)
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    if filtered_participants_file_path:
        limit_participants_df = pd.read_csv(filtered_participants_file_path, dtype={'ID': str})
        filtered_participant_ids = limit_participants_df['ID'].tolist()
    else:
        filtered_participant_ids = []
    if len(sep) != 1:
        sep = codecs.decode(sep.encode('utf8'), 'unicode_escape')

    with gzip.open(input_file_paths[0], 'rt') as f:
        if skip_rows:
            for _ in range(skip_rows):
                f.readline()
            column_line = f.readline()
        else:
            stop = False
            while not stop:
                line = f.readline()
                if line.startswith('##'):
                    skip_rows += 1
                else:
                    stop = True
                    column_line = line
        columns = column_line.strip().split(sep)
        participant_columns = list(set(columns) - set(snp_info_columns))
        if filtered_participants_file_path and filtered_participant_ids:
            participant_columns = list(set(participant_columns) & set(filtered_participant_ids))
        # participant_columns_len = len(participant_columns)
        filtered_columns = ['#CHROM', 'POS', 'ALT', *participant_columns]
        filtered_columns_index = [columns.index(column) for column in filtered_columns]
        columns_file = open(os.path.join(output_dir_path, 'columns.csv'), 'w')
        columns_file.write('Participant ID')
        participant_files = []
        for participant_id in participant_columns:
            file = open(os.path.join(output_dir_path, f'{participant_id}.csv'), 'w')
            file.write(f'{participant_id}')
            participant_files.append(file)
    for input_file_path in input_file_paths:
        print(f'正在处理文件{input_file_path}')
        count = 0
        start = time()
        with gzip.open(input_file_path, 'rt') as f:
            for _ in range(skip_rows + 1):
                f.readline()
            for line in f:
                count += 1
                columns = line.strip().split(sep)
                filtered_columns = list((itemgetter(*filtered_columns_index)(columns)))
                if regularize:
                    # 注意这里的filtered_columns的索引要和filtered_columns前面有多少列对应
                    regularized_columns = [regularize_rules[participant_column.split(':')[0]] for participant_column in
                                           filtered_columns[3:]]
                    # 所有样本中，小于某个百分比的 SNP 都不要
                    # if (sum(regularized_columns) / participant_columns_len) <= samples_ratio:
                    #     continue
                columns_file.write(f',{filtered_columns[0][3:]}:{filtered_columns[1]}')
                for index, file in enumerate(participant_files):
                    if regularize:
                        file.write(f',{regularized_columns[index]}')
                    else:
                        file.write(f',{filtered_columns[index + 3]}')
                if count % 1000 == 0:
                    columns_file.flush()
                    for file in participant_files:
                        file.flush()
                    print(f'count:{count},time:{time() - start}')
    columns_file.close()
    for file in participant_files:
        file.close()


if __name__ == '__main__':
    """
    需要注意更改是否是压缩文件，是否使用gzip模块
    """
    main()
