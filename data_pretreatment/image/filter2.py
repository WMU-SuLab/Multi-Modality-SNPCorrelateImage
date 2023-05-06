# -*- encoding: utf-8 -*-
"""
@File Name      :   filter2.py
@Create Time    :   2023/2/22 20:13
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
@click.option('-n', '--image_name_field_name', default='picture_name')
@click.option('-o', '--output_file_name', default='filtered_img_file_paths.txt')
def main(img_dir_path: str, filtered_participants_file_path: str, image_name_field_name: str, output_file_name: str):
    """
    根据原始图片文件名称，筛选出符合条件的图片文件路径
    :param img_dir_path:
    :param filtered_participants_file_path:
    :param image_name_field_name:
    :param output_file_name:
    :return:
    """
    img_dir_path = os.path.abspath(img_dir_path)
    df = pd.read_csv(filtered_participants_file_path)
    image_file_names = df[image_name_field_name].tolist()
    img_file_paths = [
        os.path.join(img_dir_path, file_name)
        for file_name in os.listdir(img_dir_path)
        if file_name in image_file_names
    ]
    with open(os.path.join(os.path.dirname(filtered_participants_file_path), output_file_name), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(img_file_paths))


if __name__ == '__main__':
    main()
