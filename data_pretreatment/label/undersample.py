# -*- encoding: utf-8 -*-
"""
@File Name      :   unsersampling.py
@Create Time    :   2023/4/11 16:00
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
def main(input_file_path: str):
    input_dir_path = os.path.dirname(input_file_path)
    input_file_name = os.path.basename(input_file_path)
    df = pd.read_csv(input_file_path, dtype={'是否高度近视-SE': int})
    myopia_df = df[df['是否高度近视-SE'] == 1]
    not_myopia_df = df[df['是否高度近视-SE'] == 0]
    times = math.ceil(not_myopia_df.shape[0] / myopia_df.shape[0])
    # 重复
    myopia_df = pd.concat([myopia_df] * times)
    # 打乱
    myopia_df = myopia_df.sample(frac=1)
    # 随机采样
    myopia_df = myopia_df.sample(n=not_myopia_df.shape[0])
    # 合并
    df = pd.concat([myopia_df, not_myopia_df])
    df.to_csv(os.path.join(input_dir_path, f'undersample_{input_file_name}'), index=False)


if __name__ == '__main__':
    main()
