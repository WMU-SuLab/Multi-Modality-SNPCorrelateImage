# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_snps_dataset_merge.py   
@Create Time    :   2024/1/30 14:42
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@Other Info     :
"""
__auth__ = 'diklios'

import os

import click
import tqdm


@click.command()
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.argument('another_dataset_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
def main(dataset_dir_path, another_dataset_dir_path, output_dir_path):
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    dataset_file_paths = [file_name for file_name in os.listdir(dataset_dir_path)
                          if file_name.endswith('csv')]
    another_dataset_file_paths = [file_name for file_name in os.listdir(another_dataset_dir_path)
                                  if file_name.endswith('csv')]
    file_paths_set = set(dataset_file_paths + another_dataset_file_paths)
    for file_path in tqdm.tqdm(file_paths_set):
        with open(os.path.join(dataset_dir_path, file_path), 'r', encoding='utf-8') as f:
            dataset_lines = f.read().split(',')
        with open(os.path.join(another_dataset_dir_path, file_path), 'r', encoding='utf-8') as f:
            another_dataset_lines = f.read().split(',')
        with open(os.path.join(output_dir_path, file_path), 'w', encoding='utf-8') as f:
            f.write(','.join(dataset_lines + another_dataset_lines[1:]))


if __name__ == '__main__':
    main()
