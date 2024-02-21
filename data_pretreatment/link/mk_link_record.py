# -*- encoding: utf-8 -*-
"""
@File Name      :   mk_link_record.py
@Create Time    :   2023/7/20 16:22
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

import click


@click.command()
@click.option('-a', '--all_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('-d', '--divide_dir_path', type=click.Path(exists=True, dir_okay=True), default=None)
@click.option('-n', '--record_file_name', type=str, default=None)
def main(all_dir_path: str, divide_dir_path: str, record_file_name: str):
    """
    根据现有的软连接构建一个记录文件，和重新构建链接不同的是，记录到文件可以直接对整个划分数据集的文件夹以及多个划分文件夹进行操作
    :param all_dir_path: 例：data/divide
    :param divide_dir_path: 例：data/divide/20230419153742
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
    if not record_file_name:
        record_file_name = 'link_record.csv'
    dataset_dir_paths = [dataset_dir_path for divide_dir_path in divide_dir_paths
                         for dir_name in os.listdir(divide_dir_path)
                         if os.path.isdir(dataset_dir_path := os.path.join(divide_dir_path, dir_name))]
    link_dir_paths = [link_dir_path for dataset_dir_path in dataset_dir_paths for link_dir_name in
                      os.listdir(dataset_dir_path)
                      if os.path.isdir(link_dir_path := os.path.join(dataset_dir_path, link_dir_name))]
    for link_dir_path in link_dir_paths:
        print(f'handling {link_dir_path}')
        texts = '\n'.join([f'{link_file_path},{os.path.realpath(link_file_path)}'
                           for file_name in os.listdir(os.path.join(link_dir_path))
                           if os.path.islink(link_file_path := os.path.join(link_dir_path, file_name))])
        if not texts:
            continue
        with open(os.path.join(link_dir_path, record_file_name), 'w') as f:
            f.write(texts)


if __name__ == '__main__':
    main()
