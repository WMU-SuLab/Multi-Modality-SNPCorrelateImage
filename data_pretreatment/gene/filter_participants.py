# -*- encoding: utf-8 -*-
"""
@File Name      :   filter.py
@Create Time    :   2022/11/25 17:43
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
from time import time

import click
import pandas as pd

from base import snp_info_columns


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_path', default='filtered_participants.csv', help='合并后的文件名')
@click.option('-f', '--filtered_participants_file_path', help='限制的参与者的文件路径')
@click.option('-c', '--chunk_size', default=1000, help='每次读取的行数')
@click.option('-s', '--skip_rows', default=27, help='跳过的行数')
def main(input_file_path: str, output_file_path: str, filtered_participants_file_path: str, chunk_size: int,
         skip_rows: int):
    input_file_path = os.path.abspath(input_file_path)
    output_file_path = os.path.abspath(output_file_path)
    if filtered_participants_file_path:
        limit_participants_df = pd.read_csv(filtered_participants_file_path, dtype={'ID': str})
        filtered_participant_ids = limit_participants_df['ID'].tolist()
    else:
        filtered_participant_ids = []

    count = 0
    filtered_columns = []
    start = time()
    for df in pd.read_csv(input_file_path, sep='\t', chunksize=chunk_size, skiprows=skip_rows):
        count += 1
        print(f'count:{count}')
        if count == 1:
            participant_columns = list(set(df.columns) - set(snp_info_columns))
            if filtered_participants_file_path and filtered_participant_ids:
                participant_columns = list(set(participant_columns) & set(filtered_participant_ids))
            filtered_columns = ['ID', *participant_columns]
        df = df[filtered_columns]
        if count == 1:
            df.to_csv(output_file_path, index=False)
        else:
            df.to_csv(output_file_path, index=False, mode='a', header=False)
        print((time() - start) / 60)


if __name__ == '__main__':
    main()
