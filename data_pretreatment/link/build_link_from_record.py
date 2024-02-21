# -*- encoding: utf-8 -*-
"""
@File Name      :   build_link_from_record.py   
@Create Time    :   2023/7/20 16:25
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@other information
"""
__auth__ = 'diklios'

import os
from pathlib import Path

import click


def is_windows_path(path):
    return "\\" in path


def remove_unused_symlinks(current_dir):
    current_dir = Path(current_dir)
    for item in current_dir.iterdir():
        if item.is_symlink() and not item.resolve().exists():
            item.unlink()


@click.command()
@click.argument('new_data_dir_path', type=click.Path(exists=True, dir_okay=True))
@click.option('-a', '--all_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('-d', '--divide_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('-n', '--record_file_name', type=str, default=None)
def main(new_data_dir_path: str, all_dir_path: str, divide_dir_path: str, record_file_name: str):
    """
    根据现有的记录文件重新构建软连接
    :param all_dir_path:
    :param divide_dir_path:
    :param new_data_dir_path:
    :return:
    """
    new_data_dir_path = os.path.abspath(new_data_dir_path)
    print(f'new_data_dir_path: {new_data_dir_path}')
    if all_dir_path and os.path.isdir(all_dir_path):
        all_dir_path = os.path.abspath(all_dir_path)
        divide_dir_paths = [divide_dir_path for dir_name in os.listdir(all_dir_path)
                            if os.path.isdir(divide_dir_path := os.path.join(all_dir_path, dir_name))]
    elif divide_dir_path and os.path.isdir(divide_dir_path):
        divide_dir_path = os.path.abspath(divide_dir_path)
        divide_dir_paths = [divide_dir_path]
    else:
        print('Please input a valid dir path')
        return
    if not record_file_name:
        record_file_name = 'link_record.csv'
    dataset_dir_paths = [dataset_dir_path for divide_dir_path in divide_dir_paths
                         for dir_name in os.listdir(divide_dir_path)
                         if os.path.isdir(dataset_dir_path := os.path.join(divide_dir_path, dir_name))]
    link_dir_paths = [link_dir_path for dataset_dir_path in dataset_dir_paths for link_dir_name in
                      os.listdir(dataset_dir_path)
                      if os.path.isdir(link_dir_path := os.path.join(dataset_dir_path, link_dir_name))]
    for link_dir_path in link_dir_paths:
        remove_unused_symlinks(link_dir_path)
        print(link_dir_path)
        link_record_path = os.path.join(link_dir_path, record_file_name)
        if not os.path.exists(link_record_path):
            continue
        with open(link_record_path, 'r') as f:
            text = f.read()
        paths = text.split('\n')
        paths = list(filter(lambda x: x and x.strip(), paths))
        if not paths:
            continue
        for path in paths:
            target_path, source_path = path.split(',')
            if is_windows_path(target_path):
                base_name = target_path.split('\\')[-1]
                new_source_path = os.path.join(new_data_dir_path, *source_path.split('\\')[-3:])
            else:
                base_name = target_path.split('/')[-1]
                new_source_path = os.path.join(new_data_dir_path, *source_path.split('/')[-3:])
            new_target_path = os.path.join(link_dir_path, base_name)
            if os.path.lexists(new_target_path) or os.path.exists(new_target_path) \
                    or os.path.islink(new_target_path) or os.path.isfile(new_target_path):
                os.remove(new_target_path)
            os.symlink(new_source_path, new_target_path)


if __name__ == '__main__':
    """
    python data_pretreatment/link/build_link_from_record.py work_dirs/data/ -a work_dirs/data/divide/
    """
    main()
