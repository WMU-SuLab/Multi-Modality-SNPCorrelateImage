# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_transpose_split_with_threads.py
@Create Time    :   2022/11/30 15:31
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
from collections import defaultdict
from threading import Thread

import click
import pandas as pd

from base import snp_info_columns


def handle_save_participants(row: pd.Series, participant_file_path: str):
    row.to_csv(participant_file_path, index=False)


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-c', '--chunk_size', default=10000, help='每次读取的行数')
@click.option('-s', '--skip_rows', default=0, help='跳过的行数')
def main(input_file_path: str, chunk_size: int, skip_rows: int):
    input_file_path = os.path.abspath(input_file_path)
    # 分
    output_dir_path = os.path.dirname(input_file_path)
    split_dir_path = os.path.join(output_dir_path, 'split')
    if not os.path.exists(split_dir_path) and not os.path.isdir(split_dir_path):
        os.mkdir(split_dir_path)
    split_count = 0
    participant_columns = []
    split_group = defaultdict(list)
    threads = []
    for chromosome_df in pd.read_csv(input_file_path, chunksize=chunk_size, skiprows=skip_rows):
        split_count += 1
        print(f'split count:{split_count}')
        if split_count == 1:
            participant_columns = list(set(chromosome_df.columns) - set(snp_info_columns))
        snp_columns = chromosome_df['ID'].tolist()
        # 对文件重新组织为以参与者为行的形式，但是比较消耗内存，而且多线程方法并没有更快，所以放弃
        new_chromosome_df = pd.DataFrame(columns=['Participant ID', *snp_columns])
        new_chromosome_df['Participant ID'] = participant_columns
        new_chromosome_df[snp_columns] = chromosome_df[participant_columns].values.T
        for index, row in new_chromosome_df.iterrows():
            participant_file_path = os.path.join(split_dir_path, f"{row['Participant ID']}_{split_count}.csv")
            split_group[row['Participant ID']].append(participant_file_path)
            new_row = row.to_frame().T
            thread = Thread(target=handle_save_participants, args=(new_row, participant_file_path))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        threads.clear()
    json.dump(split_group, open(os.path.join(output_dir_path, 'split_group.json'), 'w'))


if __name__ == '__main__':
    main()
