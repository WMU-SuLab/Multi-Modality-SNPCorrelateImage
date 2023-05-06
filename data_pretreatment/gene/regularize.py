# -*- encoding: utf-8 -*-
"""
@File Name      :   regularize.py
@Create Time    :   2022/11/30 20:33
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
import os
from time import time

import click
import pandas as pd

from base import snp_info_columns


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_path', default='regular_chromosomes.csv', help='规范化后的文件名')
@click.option('--sep', default='\t', help='分隔符')
@click.option('-c', '--chunk_size', default=10000, help='每次读取的行数')
@click.option('-s', '--skip_rows', default=28, help='跳过的行数')
@click.option('--samples_ratio', default=0.001, type=float, help='样本比例')
def main(input_file_path: str, output_file_path: str, sep: str, chunk_size: int, skip_rows: int, samples_ratio: float):
    input_file_path = os.path.abspath(input_file_path)
    output_file_path = os.path.abspath(output_file_path)
    if len(sep) != 1:
        sep = codecs.decode(sep.encode('utf8'), 'unicode_escape')
    count = 0
    participant_columns = []
    start = time()
    for df in pd.read_csv(input_file_path, sep=sep, chunksize=chunk_size, skiprows=skip_rows):
        count += 1
        print(f'count:{count}')
        if count == 1:
            participant_columns = list(set(df.columns) - set(snp_info_columns))
        participant_columns_len = len(participant_columns)
        df.replace({
            './.': 0,
            '0/0': 0,
            '0/1': 1,
            '1/0': 1,
            '1/1': 2,
        }, inplace=True)
        # 所有样本中，小于某个百分比的SNP，都不要
        df = df[~(df[participant_columns].sum(axis=1) / participant_columns_len <= samples_ratio)]
        if count == 1:
            df.to_csv(output_file_path, index=False)
        else:
            df.to_csv(output_file_path, index=False, mode='a', header=False)
        print((time() - start) / 60)
    print('Done!')


if __name__ == '__main__':
    main()
