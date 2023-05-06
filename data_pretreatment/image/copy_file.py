# -*- encoding: utf-8 -*-
"""
@File Name      :   copy.py
@Create Time    :   2022/12/6 16:53
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
from shutil import copy2

import click


@click.command()
@click.argument('filtered_img_file_path', type=click.Path(exists=True))
@click.argument('target_dir_path', type=click.Path(exists=True))
def main(filtered_img_file_path: str, target_dir_path: str):
    """
    将筛选出来的图片文件复制到指定文件夹中
    :param filtered_img_file_path:
    :param target_dir_path:
    :return:
    """
    with open(filtered_img_file_path, 'r', encoding='utf-8-sig') as f:
        img_file_paths = f.read().splitlines()
    for img_file_path in img_file_paths:
        if img_file_path and os.path.exists(img_file_path):
            copy2(img_file_path, target_dir_path)


if __name__ == '__main__':
    """
    由于和copy模块名称冲突，所以文件名使用copy_file.py
    """
    main()
