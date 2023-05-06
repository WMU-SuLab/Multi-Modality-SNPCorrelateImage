# -*- encoding: utf-8 -*-
"""
@File Name      :   transform.py
@Create Time    :   2022/11/1 14:53
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

from torchvision import transforms

base_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gene_image_transforms = {
    # compose是操作的集合
    'train': transforms.Compose([
        # 一般先resize，大多数是256*256
        transforms.Resize(256),
        # 从中心开始裁剪，裁剪出224x224的图片（VGG、ResNet等都是224x224）
        transforms.CenterCrop(224),
        # 也可以使用随机裁剪，得到的种类更多
        # transforms.RandomCrop(224),
        # 随机旋转，-30到30度之间随机旋转
        # transforms.RandomRotation(30),
        # 随机缩放，0.5到1.0之间随机缩放
        # transforms.RandomResizedCrop(224),
        # 随机水平翻转，一半的概率水平翻转，一半的概率不翻转
        # transforms.RandomHorizontalFlip(),
        # 随机垂直翻转，一半的概率垂直翻转，一半的概率不翻转
        # transforms.RandomVerticalFlip(),
        # 亮度，对比度，饱和度，色相
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # 概率转化为灰度图，但是仍然保留三个通道，不然无法训练
        # transforms.RandomGrayscale(p=0.1),
        # 转换为Tensor
        transforms.ToTensor(),
        # 归一化均值和标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
