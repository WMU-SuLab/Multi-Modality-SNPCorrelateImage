# -*- encoding: utf-8 -*-
"""
@File Name      :   speed_up_with_chosen_snps.py
@Create Time    :   2023/3/29 14:58
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
from collections import defaultdict
from operator import itemgetter
from time import time

import click
import pandas as pd

from base import snp_info_columns

regularize_rules = {
    './.': 0,
    '0/0': 0,
    '0/1': 1,
    '1/0': 1,
    '1/1': 2,
    '0/2': 2,
}


@click.command()
@click.argument('input_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
@click.argument('chosen_snps_file_path', type=click.Path(exists=True))
@click.option('-f', '--filtered_participants_file_path', help='限制的参与者的文件路径')
@click.option('-s', '--skip_rows', default=0, help='跳过的行数')
@click.option('--sep', default='\t', help='分隔符')
def main(input_dir_path: str, output_dir_path: str, chosen_snps_file_path: str, filtered_participants_file_path: str,
         skip_rows: int, sep: str):
    input_dir_path = os.path.abspath(input_dir_path)
    input_file_names = [file_name for file_name in os.listdir(input_dir_path) if file_name.endswith('.vcf.gz')]
    input_file_paths = [os.path.join(input_dir_path, file_name) for file_name in input_file_names]
    chosen_snps_file_path = os.path.abspath(chosen_snps_file_path)
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    # 前期处理
    if filtered_participants_file_path:
        limit_participants_df = pd.read_csv(filtered_participants_file_path, dtype={'ID': str})
        filtered_participant_ids = limit_participants_df['ID'].tolist()
    else:
        filtered_participant_ids = []
    snps_df = pd.read_csv(chosen_snps_file_path)
    snps_dict = defaultdict(str)
    for index, row in snps_df.iterrows():
        # snps_dict[f"{row['CHR']}:{row['BP']}"] = row['A1']
        snps_dict[f"{row['chromosome']}:{row['base_pair_location']}"] = row['effect_allele']
    if len(sep) != 1:
        sep = codecs.decode(sep.encode('utf8'), 'unicode_escape')
    # 创建文件
    with gzip.open(input_file_paths[0], 'rt') as f:
        column_line = ''
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
        filtered_columns = ['#CHROM', 'POS', 'ALT', *participant_columns]
        filtered_columns_index = [columns.index(filtered_column) for filtered_column in filtered_columns]
        columns_file = open(os.path.join(output_dir_path, 'columns.csv'), 'w')
        columns_file.write('Participant ID')
        participant_files = []
        for participant_column in participant_columns:
            file = open(os.path.join(output_dir_path, f'{participant_column}.csv'), 'w')
            file.write(f'{participant_column}')
            participant_files.append(file)
    # 读取文件
    for input_file_path in input_file_paths:
        print(f'正在处理文件{input_file_path}')
        count = 0
        start = time()
        with gzip.open(input_file_path, 'rt') as f:
            if skip_rows + 1:
                for _ in range(skip_rows):
                    f.readline()
            for line in f:
                columns = line.strip().split(sep)
                filtered_columns = list((itemgetter(*filtered_columns_index)(columns)))
                snp_id = f"{filtered_columns[0][3:]}:{filtered_columns[1]}"
                if snps_dict[snp_id]:
                    count += 1
                else:
                    continue
                regularized_columns = [regularize_rules[participant_column.split(':')[0]] for participant_column
                                       in filtered_columns[3:]]
                if snps_dict[snp_id] != filtered_columns[2]:
                    regularized_columns = [2 - regularized_column for regularized_column in regularized_columns]
                columns_file.write(f',{snp_id}')
                for index, file in enumerate(participant_files):
                    file.write(f',{regularized_columns[index]}')
                if count % 100 == 0:
                    columns_file.flush()
                    for file in participant_files:
                        file.flush()
                    print(f'row count:{count}, time:{time() - start}')
    # 关闭文件
    columns_file.close()
    for file in participant_files:
        file.close()


if __name__ == '__main__':
    main()
