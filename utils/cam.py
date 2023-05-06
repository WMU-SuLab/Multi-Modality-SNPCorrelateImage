# -*- encoding: utf-8 -*-
"""
@File Name      :   cam.py
@Create Time    :   2023/4/10 18:30
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

import matplotlib.pyplot as plt
import pytorch_grad_cam as cams
from pytorch_grad_cam.utils.image import show_cam_on_image

from .img import tensor2numpy
from .time import datetime_now


def imshow(images, titles=None, file_name="test.jpg", size=6):
    lens = len(images)
    fig = plt.figure(figsize=(size * lens, size))
    if not titles:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(images[i - 1].shape) == 2:
            plt.imshow(images[i - 1], cmap='Reds')
        else:
            plt.imshow(images[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(f'{datetime_now()}_{file_name}', bbox_inches='tight')


class ReshapeTransform:
    def __init__(self, height: int = 14, width: int = 14):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        result = tensor.reshape(tensor.size(0), self.height, self.width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result


def make_cam(cam_name, net, target_layers: list, targets: list, img_tensor, use_cuda=False):
    """
    :param cam_name:
    :param net:
    :param target_layers: 如果传入多个layer，cam输出结果将会取均值
    :param targets:
    :param img_tensor: 需要是经过transform的图
    :param use_cuda:
    :return:
    """
    input_tensor = img_tensor.unsqueeze(0)
    # 如果传入多个layer，cam输出结果将会取均值
    # reshape_transform 是当输出的feature map不是不是 BCHW 的形式时，需要进行reshape
    if not (CAM := getattr(cams, cam_name, None)):
        raise ValueError(f'{cam_name} does not exist.')
    cam = CAM(model=net, target_layers=target_layers, use_cuda=use_cuda)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # targets=None 自动调用概率最大的类别显示
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
    # grayscale_cams 根据 targets 的数量，会返回多个热力图
    # for grayscale_cam in grayscale_cams:
    grayscale_cam = grayscale_cams[0, :]
    img = tensor2numpy(img_tensor)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return img, grayscale_cam, visualization
