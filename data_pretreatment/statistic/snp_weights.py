# -*- encoding: utf-8 -*-
"""
@File Name      :   snp_weights.py   
@Create Time    :   2024/1/7 20:36
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

import click
import matplotlib.pyplot as plt
import pandas as pd


@click.command()
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('method', type=click.Choice(['num', 'percent']))
@click.argument('threshold_right', type=float)
@click.option('--threshold_left', type=float, default=0)
@click.option('--select_by', type=click.Choice(['weight_abs', 'weight', 'weight_plus', 'weight_minus']),
              default='weight_abs')
@click.option('--ascending', is_flag=True, default=False)
def main(wts_path, method, threshold_right, threshold_left, select_by, ascending):
    df = pd.read_csv(wts_path)
    df = df.sort_values(select_by, ascending=ascending)
    if threshold_left > threshold_right:
        raise ValueError('threshold_left must be less than threshold_right')
    if method == 'num':
        df = df.iloc[round(threshold_left):round(threshold_right)]
    elif method == 'percent':
        df = df.iloc[round(len(df) * threshold_left):round(len(df) * threshold_right)]
    else:
        raise ValueError('method must be num or percent')
    x = df['gene_id'].tolist()
    x += ['.', '..', '...']
    y = df[select_by].tolist()
    print(y)
    y += [0, 0, 0]
    x = list(reversed(x))
    y = list(reversed(y))
    plt.barh(x, y, color='#FF0050', edgecolor='black', linewidth=1)
    plt.xlabel(select_by)
    plt.ylabel('SNP')
    plt.title('SNP Weights')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
