# -*- encoding: utf-8 -*-
"""
@File Name      :   screen.py
@Create Time    :   2022/11/29 11:31
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
from datetime import datetime

import click
import pandas as pd

from base import data_dir, AMD_file_path, DR_file_path, GLC_file_path, RD_file_path, no_disease_file_path


@click.command()
@click.argument('disease', type=click.Choice(['AMD', 'DR', 'GLC', 'RD']))
@click.option('-n', '--no_disease_num', default=1000, help='The number of no disease participants.')
def main(disease: str, no_disease_num: int):
    if disease == 'AMD':
        disease_file_path = AMD_file_path
    elif disease == 'DR':
        disease_file_path = DR_file_path
    elif disease == 'GLC':
        disease_file_path = GLC_file_path
    elif disease == 'RD':
        disease_file_path = RD_file_path
    else:
        raise ValueError('The disease name is not correct.')
    disease_df = pd.read_csv(disease_file_path, sep='\t')
    no_disease_df = pd.read_csv(no_disease_file_path, sep='\t')
    no_disease_df = no_disease_df.sample(n=no_disease_num)
    df = pd.concat([disease_df, no_disease_df])
    df.to_csv(
        os.path.join(
            data_dir,
            f"{disease}_{no_disease_num}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_screen.csv"
        ),
        sep='\t',
        index=False
    )


if __name__ == '__main__':
    main()
