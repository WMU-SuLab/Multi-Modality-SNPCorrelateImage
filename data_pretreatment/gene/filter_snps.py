# -*- encoding: utf-8 -*-
"""
@File Name      :   filter_snp.py
@Create Time    :   2023/3/10 16:46
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


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_path', default='filtered_snps.csv', help='合并后的文件名')
@click.option('--or_value', default=1, type=float, help='OR')
@click.option('-p', '--p_value', default=0.05, type=float, help='p值')
def main(input_file_path: str, output_file_path: str, or_value: float, p_value: float):
    input_file_path = os.path.abspath(input_file_path)
    df = pd.read_csv(input_file_path, sep='\t')
    # df = df[(df['OR'] > or_value) & (df['MLMALOCO'] < p_value)]
    df = df[df['MLMALOCO'] < p_value]
    df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    main()
