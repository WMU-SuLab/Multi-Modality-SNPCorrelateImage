# -*- encoding: utf-8 -*-
"""
@File Name      :   snp_weights_compare.py   
@Create Time    :   2024/2/17 17:46
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
import pandas as pd


@click.command()
@click.argument('compared_columns_file_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('wts_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('method', type=click.Choice(['num', 'percent']))
@click.argument('threshold_right', type=float)
@click.option('--threshold_left', type=float, default=0)
def main(compared_columns_file_path, wts_path, method, threshold_right, threshold_left):
    with open(compared_columns_file_path, 'r') as f:
        compared_snps = f.read().split(',')
    print(f'compared_columns:{len(compared_snps)}')
    compared_snps = set(compared_snps)
    df = pd.read_csv(wts_path)
    df_weight = df.copy().sort_values('weight',ascending=False)
    df_weight_abs = df.copy().sort_values('weight_abs',ascending=False)
    df_weight_plus = df.copy().sort_values('weight_plus',ascending=False)
    df_weight_minus = df.copy().sort_values('weight_minus',ascending=False)
    if threshold_left > threshold_right:
        raise ValueError('threshold_left must be less than threshold_right')
    if method == 'num':
        df_weight = df_weight.iloc[round(threshold_left):round(threshold_right)]
        df_weight_abs = df_weight_abs.iloc[round(threshold_left):round(threshold_right)]
        df_weight_plus = df_weight_plus.iloc[round(threshold_left):round(threshold_right)]
        df_weight_minus = df_weight_minus.iloc[round(threshold_left):round(threshold_right)]
    elif method == 'percent':
        df_weight = df_weight.iloc[round(len(df) * threshold_left):round(len(df) * threshold_right)]
        df_weight_abs = df_weight_abs.iloc[round(len(df) * threshold_left):round(len(df) * threshold_right)]
        df_weight_plus = df_weight_plus.iloc[round(len(df) * threshold_left):round(len(df) * threshold_right)]
        df_weight_minus = df_weight_minus.iloc[round(len(df) * threshold_left):round(len(df) * threshold_right)]
    else:
        raise ValueError('method must be num or percent')
    snps_weight = df_weight['gene_id'].tolist()
    snps_weight_abs = df_weight_abs['gene_id'].tolist()
    snps_weight_plus = df_weight_plus['gene_id'].tolist()
    snps_weight_minus = df_weight_minus['gene_id'].tolist()
    snps=df['gene_id'].tolist()
    # print(f'snps:{len(snps)}')
    # print(f'snps_weight:{len(snps_weight)}')
    # print(f'snps_weight_abs:{len(snps_weight_abs)}')
    # print(f'snps_weight_plus:{len(snps_weight_plus)}')
    # print(f'snps_weight_minus:{len(snps_weight_minus)}')

    snps_set = set(snps) & compared_snps
    print(f'snps_set:{len(snps_set)}')
    # print(snps_set)
    snps_set_weight = set(snps_weight) & compared_snps
    print(f'snps_set_weight:{len(snps_set_weight)}')
    print(snps_set_weight)
    snps_set_weight_abs = set(snps_weight_abs) & compared_snps
    print(f'snps_set_weight_abs:{len(snps_set_weight_abs)}')
    print(snps_set_weight_abs)
    snps_set_weight_plus = set(snps_weight_plus) & compared_snps
    print(f'snps_set_weight_plus:{len(snps_set_weight_plus)}')
    print(snps_set_weight_plus)
    snps_set_weight_minus = set(snps_weight_minus) & compared_snps
    print(f'snps_set_weight_minus:{len(snps_set_weight_minus)}')
    print(snps_set_weight_minus)

    snps_set_and = snps_set_weight & snps_set_weight_abs & snps_set_weight_plus
    print(f'snps_set_and:{len(snps_set_and)}')
    print(snps_set_and)
    snps_set_or = snps_set_weight | snps_set_weight_abs | snps_set_weight_plus | snps_set_weight_minus
    print(f'snps_set_or:{len(snps_set_or)}')
    print(snps_set_or)


if __name__ == '__main__':
    main()
