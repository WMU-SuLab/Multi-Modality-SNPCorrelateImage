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


@click.command()
@click.option('-a', '--all_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('-d', '--divide_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('-n', '--new_data_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
def main(all_dir_path: str, divide_dir_path: str, new_data_dir_path: str):
    """
    根据现有的记录文件重新构建软连接
    :param all_dir_path:
    :param divide_dir_path:
    :param new_data_dir_path:
    :return:
    """
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
    dataset_dir_paths = [dataset_dir_path for divide_dir_path in divide_dir_paths
                         for dir_name in os.listdir(divide_dir_path)
                         if os.path.isdir(dataset_dir_path := os.path.join(divide_dir_path, dir_name))]
    link_dir_paths = [link_dir_path for dataset_dir_path in dataset_dir_paths for link_dir_name in
                      os.listdir(dataset_dir_path)
                      if os.path.isdir(link_dir_path := os.path.join(dataset_dir_path, link_dir_name))]
    for link_dir_path in link_dir_paths:
        print(link_dir_path)
        link_record_path = os.path.join(link_dir_path, 'link_record.csv')
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
            base_name = os.path.basename(target_path)
            new_target_path = os.path.join(link_dir_path, base_name)
            if new_data_dir_path:
                new_source_path = os.path.join(new_data_dir_path, *Path(source_path).parts[-3:])
            else:
                new_source_path = source_path
            if os.path.lexists(new_target_path) or os.path.exists(new_target_path) \
                    or os.path.islink(new_target_path) or os.path.isfile(new_target_path):
                os.remove(new_target_path)
            print(new_source_path, new_target_path)
            os.symlink(new_source_path, new_target_path)


if __name__ == '__main__':
    main()
