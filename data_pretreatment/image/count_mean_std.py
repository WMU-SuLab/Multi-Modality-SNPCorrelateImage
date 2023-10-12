# -*- encoding: utf-8 -*-
"""
@File Name      :   count_mean_std.py
@Create Time    :   2023/8/1 17:13
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
import cv2
import numpy as np


@click.command()
@click.argument('image_dir_path', type=click.Path(exists=True, dir_okay=True, file_okay=False))
def main(image_dir_path):
    """
    计算图片文件夹中所有图片的均值和标准差
    :param image_dir_path:
    :return:
    """
    image_dir_path = os.path.abspath(image_dir_path)
    image_file_path = [file_path for file_name in os.listdir(image_dir_path)
                       if os.path.isfile(file_path := os.path.join(image_dir_path, file_name))
                       and (file_name.endswith('.jpg') or file_name.endswith('.png'))]
    num = len(image_file_path)
    mean_r = 0
    mean_g = 0
    mean_b = 0
    sd_r = 0
    sd_g = 0
    sd_b = 0
    for image_path in image_file_path:
        image = cv2.imread(image_path)
        mean_b += np.mean(image[:, :, 0])
        mean_g += np.mean(image[:, :, 1])
        mean_r += np.mean(image[:, :, 2])
    mean_b /= num
    mean_g /= num
    mean_r /= num
    for image_path in image_file_path:
        image = cv2.imread(image_path)
        sd_b += (np.mean(image[:, :, 0]) - mean_b) ** 2
        sd_g += (np.mean(image[:, :, 1]) - mean_g) ** 2
        sd_r += (np.mean(image[:, :, 2]) - mean_r) ** 2
    sd_b = (sd_b / num) ** 0.5
    sd_g = (sd_g / num) ** 0.5
    sd_r = (sd_r / num) ** 0.5
    print(f"mean_b:{mean_b},mean_g:{mean_g},mean_r:{mean_r}")
    print(f"sd_b:{sd_b},sd_g:{sd_g},sd_r:{sd_r}")


if __name__ == '__main__':
    main()
