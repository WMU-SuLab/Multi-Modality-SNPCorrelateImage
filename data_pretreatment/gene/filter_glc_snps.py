# -*- encoding: utf-8 -*-
"""
@File Name      :   filter_glc.py
@Create Time    :   2023/5/22 16:52
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

import math
import os

import click
import pandas as pd


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_name', default='', help='合并后的文件名')
@click.option('-p', '--p_value', default=0, type=float, help='p值')
@click.option('--or_value', default=0, type=float, help='OR')
def main(input_file_path: str, output_file_name: str, p_value: float, or_value: float, ):
    input_file_path = os.path.abspath(input_file_path)
    input_dir = os.path.dirname(input_file_path)
    if not output_file_name:
        output_file_name = os.path.basename(input_file_path)
    df = pd.read_csv(input_file_path, sep='\t')
    df['OR'] = math.exp(df['beta'])
    if or_value and p_value:
        df = df[(df['OR'] > or_value) & (df['p_value'] < p_value)]
    elif p_value:
        df = df[df['p_value'] < p_value]
    elif or_value:
        df = df[df['OR'] > or_value]
    else:
        raise ValueError('请至少输入一个筛选条件')
    print(df.shape)
    df.to_csv(os.path.join(input_dir, f'p_value_{p_value}_OR_{or_value}_{output_file_name}'), index=False)
