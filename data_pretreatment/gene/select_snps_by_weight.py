# -*- encoding: utf-8 -*-
"""
@File Name      :   select_snps_by_weight.py   
@Create Time    :   2024/1/7 20:08
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@Other Info     :
"""
__auth__ = 'diklios'

import os

import click
import pandas as pd


@click.command()
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('method', type=click.Choice(['num', 'percent']))
@click.argument('threshold', type=float)
@click.option('--select_by', '-s', type=click.Choice(['weight_abs', 'weight']), default='weight_abs')
@click.option('--ascending', '-r', is_flag=True, default=False)
@click.option('--output_dir_path', '-o', type=click.Path(dir_okay=True), default='')
def main(wts_path, method, threshold, select_by, ascending, output_dir_path):
    df = pd.read_csv(wts_path)
    df = df.sort_values(ascending=ascending, by=select_by)

    if method == 'num':
        df = df.iloc[:round(threshold)]
    elif method == 'percent':
        if threshold < 0 or threshold > 1:
            raise ValueError('threshold must between 0 and 1')
        df = df.iloc[:round(len(df) * threshold)]
    else:
        raise ValueError('method must be num or percent')
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_dir_path = os.path.join(
        output_dir_path, f"{method}_{threshold}_{select_by}_{'ascending' if ascending else 'descending'}")
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    with open(os.path.join(output_dir_path, 'chosen_columns.csv'), 'w') as f:
        f.write(','.join(df['gene_id'].tolist()))


if __name__ == '__main__':
    main()
