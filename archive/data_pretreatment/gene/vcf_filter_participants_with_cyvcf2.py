# -*- encoding: utf-8 -*-
"""
@File Name      :   filter_with_cyvcf2.py
@Create Time    :   2022/12/1 21:18
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

import click
import pandas as pd
from cyvcf2 import VCF


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_path', default='filtered_participants.csv', help='合并后的文件名')
@click.option('-f', '--filtered_participants_file_path', help='限制的参与者的文件路径')
def main(input_file_path: str, output_file_path: str, filtered_participants_file_path: str):
    input_file_path = os.path.abspath(input_file_path)
    output_file_path = os.path.abspath(output_file_path)
    if filtered_participants_file_path:
        limit_participants_df = pd.read_csv(filtered_participants_file_path, dtype={'ID': str})
        filtered_participant_ids = limit_participants_df['ID'].tolist()
        reader = VCF(input_file_path, lazy=True, samples=filtered_participant_ids)
    else:
        reader = VCF(input_file_path, lazy=True)
    with open(output_file_path, 'w') as w:
        w.write(reader.raw_header)
        for variant in reader:
            # 自带换行符，不需要再增加
            w.write(str(variant))


if __name__ == '__main__':
    main()
