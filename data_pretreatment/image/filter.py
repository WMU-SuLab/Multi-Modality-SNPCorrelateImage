# -*- encoding: utf-8 -*-
"""
@File Name      :   filter.py
@Create Time    :   2022/12/6 16:22
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
@click.argument('img_dir_path', type=click.Path(exists=True))
@click.argument('filtered_participants_file_path', type=click.Path(exists=True))
@click.option('-n', '--id_field_name', default='Participant ID')
@click.option('-o', '--output_file_name', default='filtered_img_file_paths.txt')
def main(img_dir_path: str, filtered_participants_file_path: str, id_field_name: str, output_file_name: str):
    """
    根据病人id定义的图片文件名，筛选出符合条件的图片文件路径
    :param img_dir_path:
    :param filtered_participants_file_path:
    :param id_field_name:
    :param output_file_name:
    :return:
    """
    img_dir_path = os.path.abspath(img_dir_path)
    df = pd.read_csv(filtered_participants_file_path)
    participant_ids = df[id_field_name].tolist()
    img_file_paths = [
        os.path.join(img_dir_path, file_name)
        for file_name in os.listdir(img_dir_path)
        if os.path.splitext(file_name)[0].split('_')[0] in participant_ids
    ]
    with open(os.path.join(os.path.dirname(filtered_participants_file_path), output_file_name), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(img_file_paths))


if __name__ == '__main__':
    main()
