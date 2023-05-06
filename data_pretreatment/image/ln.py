# -*- encoding: utf-8 -*-
"""
@File Name      :   ln_images.py
@Create Time    :   2022/12/19 17:43
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


@click.command()
@click.argument('img_dir_path', type=click.Path(exists=True))
@click.argument('target_img_dir_path', type=click.Path(exists=True))
def main(img_dir_path: str, target_img_dir_path: str):
    """
    给定一个图片文件夹，将其中的图片文件创建软链接到另一个文件夹中
    :param img_dir_path:
    :param target_img_dir_path:
    :return:
    """
    img_dir_path = os.path.abspath(img_dir_path)
    target_img_dir_path = os.path.abspath(target_img_dir_path)
    img_file_names = [
        file_name
        for file_name in os.listdir(img_dir_path)
        if file_name.endswith('.png')
    ]
    for img_file_name in img_file_names:
        os.symlink(os.path.join(img_dir_path, img_file_name), os.path.join(target_img_dir_path, img_file_name))
    print('Done!')


if __name__ == '__main__':
    main()
