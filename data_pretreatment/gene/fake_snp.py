# -*- encoding: utf-8 -*-
"""
@File Name      :   fake.py
@Create Time    :   2023/2/23 21:26
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
import random

import click


@click.command()
@click.argument('image_dir_path', type=click.Path(exists=True))
@click.argument('snp_dir_path', type=click.Path(exists=True))
@click.argument('snp_number', type=click.INT)
def main(image_dir_path: str, snp_dir_path: str, snp_number: int):
    image_file_names = [file_name for file_name in os.listdir(image_dir_path)
                        if file_name.endswith('.png') or file_name.endswith('.jpg')]
    fake_snp_file_paths = [os.path.join(snp_dir_path, f'{os.path.splitext(file_name)[0]}.txt')
                           for file_name in image_file_names]

    for fake_snp_file_path in fake_snp_file_paths:
        with open(fake_snp_file_path, 'w', encoding='utf-8') as f:
            f.write(' '.join([str(random.randint(0, 2)) for i in range(snp_number)]))


if __name__ == '__main__':
    main()
