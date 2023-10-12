# -*- encoding: utf-8 -*-
"""
@File Name      :   id_to_barcode.py
@Create Time    :   2023/4/11 9:25
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
@click.argument('dir_path', type=click.Path(exists=True))
def main(input_file_path, dir_path):
    df = pd.read_csv(input_file_path, dtype={'学籍号': str, '条形码': str})
    replace_dict = {row['条形码']: row['学籍号'] for index, row in df.iterrows()}
    for filename in os.listdir(dir_path):
        sp = os.path.splitext(filename)
        replace_prefix = replace_dict.get(sp[0], None)
        if replace_prefix:
            new_filename = f"{replace_prefix}{','.join(sp[1:])}"
            raw_file_path = os.path.join(dir_path, filename)
            new_file_path = os.path.join(dir_path, new_filename)
            if not os.path.exists(new_file_path):
                os.rename(raw_file_path, new_file_path)


if __name__ == '__main__':
    main()
