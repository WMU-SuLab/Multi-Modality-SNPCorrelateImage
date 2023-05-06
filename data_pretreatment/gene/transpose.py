# -*- encoding: utf-8 -*-
"""
@File Name      :   transpose.py
@Create Time    :   2022/12/19 21:45
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
from operator import itemgetter
from time import time

import click

from base import snp_info_columns


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
@click.option('-s', '--skip_rows', default=0, help='跳过的行数')
def main(input_file_path: str, output_dir_path: str, skip_rows: int):
    input_file_path = os.path.abspath(input_file_path)
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    count = 0
    start = time()
    with open(input_file_path, 'r') as f:
        if skip_rows:
            for _ in range(skip_rows):
                f.readline()
        for line in f:
            count += 1
            columns = line.strip().split(',')
            if count == 1:
                participant_columns = list(set(columns) - set(snp_info_columns))
                filtered_columns = ['ID', *participant_columns]
                filtered_columns_index = [columns.index(column) for column in filtered_columns]
                columns_file = open(os.path.join(output_dir_path, 'columns.csv'), 'w')
                columns_file.write('Participant ID')
                participant_files = []
                for participant_id in participant_columns:
                    file = open(os.path.join(output_dir_path, f'{participant_id}.csv'), 'w')
                    file.write(f'{participant_id}')
                    file.flush()
                    participant_files.append(file)
            else:
                filtered_columns = list((itemgetter(*filtered_columns_index)(columns)))
                columns_file.write(f',{filtered_columns[0]}')
                for index, file in enumerate(participant_files):
                    file.write(f',{filtered_columns[index + 1]}')
                    file.flush()
            if count % 10000 == 0:
                print(f'count:{count},time:{time() - start}')
    columns_file.close()
    for file in participant_files:
        file.close()


if __name__ == '__main__':
    main()
