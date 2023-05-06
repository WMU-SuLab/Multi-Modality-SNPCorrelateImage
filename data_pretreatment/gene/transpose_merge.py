# -*- encoding: utf-8 -*-
"""
@File Name      :   transpose_merge.py
@Create Time    :   2022/12/7 12:52
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
from functools import reduce

import click
import pandas as pd


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_path', default='transposed_chromosomes.csv', help='合并后的文件名')
def main(input_file_path: str, output_file_path: str):
    input_file_path = os.path.abspath(input_file_path)
    output_file_path = os.path.abspath(output_file_path)
    output_dir_path = os.path.dirname(output_file_path)
    # 合
    split_group = json.load(open(input_file_path, 'r'))
    participant_dir_path = os.path.join(output_dir_path, 'merge')
    participant_count = 0
    for participant_id, file_paths in split_group.items():
        participant_count += 1
        print(f'participant split_count:{participant_count}')
        participant_dfs = [pd.read_csv(file_path) for file_path in file_paths]
        df = reduce(lambda left, right: pd.merge(left, right, on=['Participant ID'], how='outer'), participant_dfs)
        df.to_csv(os.path.join(participant_dir_path, f'{participant_id}.csv'), index=False, header=False)
        if participant_count == 1:
            df.to_csv(output_file_path, index=False)
            df = df.drop(df.index)
            df.to_csv(os.path.join(participant_dir_path, 'columns.csv'), index=False)
        else:
            df.to_csv(output_file_path, index=False, mode='a', header=False)


if __name__ == '__main__':
    main()
