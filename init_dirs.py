# -*- encoding: utf-8 -*-
"""
@File Name      :   init.py
@Create Time    :   2022/10/24 16:44
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

from base import project_dir_path, root_dir_paths


@click.command()
@click.option('-p', '--parent_dir_path', default=project_dir_path, help='root_dir_path')
def init_dirs(parent_dir_path: str):
    for dir_path in root_dir_paths:
        if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
            os.mkdir(os.path.join(parent_dir_path, dir_path))
            print(f'{dir_path}文件夹创建成功')
        else:
            print(f'{dir_path}文件夹已存在')


if __name__ == '__main__':
    init_dirs()
