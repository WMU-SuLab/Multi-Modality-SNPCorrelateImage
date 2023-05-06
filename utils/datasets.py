# -*- encoding: utf-8 -*-
"""
@File Name      :   data_loader.py
@Create Time    :   2022/10/31 16:34
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

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .gene import handle_gene_file
from .name import get_gene_file_participant_id
from .participants import intersection_participants_data
from .transforms import base_image_transforms


class SNPDataset(Dataset):
    # 豹纹斑分级标签
    # label_data_id_field_name = 'picture_name'
    # label_data_label_field_name = 'tf_grade'
    # 高度近视分级标签
    label_data_id_field_name = '学籍号'
    # label_data_label_field_name = '是否高度近视-SE'
    label_data_label_field_name = 'high_myopia'

    def __init__(self, label_data_path: str, gene_data_dir_path: str,
                 label_data_id_field_name: str = None, label_data_label_field_name: str = None):
        self.label_data_path = label_data_path
        self.gene_data_dir_path = gene_data_dir_path
        if label_data_id_field_name:
            self.label_data_id_field_name = label_data_id_field_name
        if label_data_label_field_name:
            self.label_data_label_field_name = label_data_label_field_name

        self.label_df, self.gene_data_file_names = intersection_participants_data(
            label_data_path, gene_data_dir_path=gene_data_dir_path,
            label_data_id_field_name=self.label_data_id_field_name,
        )
        self.gene_data_file_paths_dict = {
            get_gene_file_participant_id(file_name): os.path.join(self.gene_data_dir_path, file_name)
            for file_name in self.gene_data_file_names}

    def get_gene_data(self, label_participant_id):
        gene_file_path = self.gene_data_file_paths_dict[label_participant_id]
        return handle_gene_file(gene_file_path)

    def __getitem__(self, index):
        label_participant_id = self.label_df.loc[index, self.label_data_id_field_name]
        gene_data = self.get_gene_data(label_participant_id)
        label_data = self.label_df.loc[index, self.label_data_label_field_name]
        return [gene_data], torch.tensor([label_data], dtype=torch.float32)

    def __len__(self):
        return self.label_df.shape[0]


class ImageDataset(Dataset):
    label_data_id_field_name = '学籍号'
    label_data_label_field_name = 'high_myopia'

    def __init__(self, label_data_path: str, image_data_dir_path: str,
                 transform=None, label_data_id_field_name: str = None, label_data_label_field_name: str = None):
        self.label_data_path = label_data_path
        self.image_data_dir_path = image_data_dir_path
        self.transform = transform if transform else base_image_transforms
        if label_data_id_field_name:
            self.label_data_id_field_name = label_data_id_field_name
        if label_data_label_field_name:
            self.label_data_label_field_name = label_data_label_field_name
        self.label_df, self.image_data_file_names = intersection_participants_data(
            label_data_path, image_data_dir_path=image_data_dir_path,
            label_data_id_field_name=self.label_data_id_field_name,
        )
        self.label_data_dict = {}
        self.set_label_data_dict()

    def set_label_data_dict(self):
        for index, row in self.label_df.iterrows():
            if (label_data := row[f'OS_{self.label_data_label_field_name}']) != -1:
                self.label_data_dict[f"{row[self.label_data_id_field_name]}_OS"] = label_data
            if (label_data := row[f'OD_{self.label_data_label_field_name}']) != -1:
                self.label_data_dict[f"{row[self.label_data_id_field_name]}_OD"] = label_data

    def get_image_label_data(self, image_file_name):
        label_participant_id, eye_side = image_file_name.split('.')[0].split('_')[0:2]
        label_data = self.label_data_dict[f'{label_participant_id}_{eye_side.upper()}']
        image = Image.open(os.path.join(self.image_data_dir_path, image_file_name))
        # 图片的归一化和标准化应该在transforms中完成
        image = self.transform(image)
        return label_participant_id, image, label_data

    def __getitem__(self, index):
        label_participant_id, image, label_data = self.get_image_label_data(self.image_data_file_names[index])
        return (image,), torch.tensor([label_data], dtype=torch.float32)

    def __len__(self):
        return len(self.image_data_file_names)


class SNPImageDataset(SNPDataset, ImageDataset):
    def __init__(self, label_data_path: str, gene_data_dir_path: str, image_data_dir_path: str,
                 transform: Compose = None, label_data_id_field_name: str = None):
        self.label_data_path = label_data_path
        self.gene_data_dir_path = gene_data_dir_path
        self.image_data_dir_path = image_data_dir_path
        self.transform = transform if transform else base_image_transforms
        if label_data_id_field_name:
            self.label_data_id_field_name = label_data_id_field_name

        self.label_df, self.gene_data_file_names, self.image_data_file_names = intersection_participants_data(
            label_data_path, gene_data_dir_path=gene_data_dir_path, image_data_dir_path=image_data_dir_path,
            label_data_id_field_name=self.label_data_id_field_name,
        )
        self.gene_data_file_paths_dict = {
            get_gene_file_participant_id(file_name): os.path.join(self.gene_data_dir_path, file_name)
            for file_name in self.gene_data_file_names}
        self.label_data_dict = {}
        self.set_label_data_dict()

    def __getitem__(self, index):
        label_participant_id, image, label_data = self.get_image_label_data(self.image_data_file_names[index])
        gene_data = self.get_gene_data(label_participant_id)
        return (gene_data, image), torch.tensor([label_data], dtype=torch.float32)

    def __len__(self):
        return ImageDataset.__len__(self)
