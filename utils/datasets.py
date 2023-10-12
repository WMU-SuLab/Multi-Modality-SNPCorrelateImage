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

import json
import os

import pandas as pd
import torch
from PIL import Image
from keras.utils import pad_sequences
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transformers import BertTokenizer

from .gene import handle_gene_file, handle_gene_file_bert
from .participants import intersection_participants_data
from .text_handler.name import get_gene_file_participant_id
from .transforms import base_image_transforms


class SNPDataset(Dataset):
    # 豹纹斑分级标签
    # label_data_id_field_name = 'picture_name'
    # label_data_label_field_name = 'tf_grade'
    # 高度近视分级标签
    label_data_id_field_name = '学籍号'
    # label_data_label_field_name = '是否高度近视-SE'
    label_data_label_field_name = 'high_myopia'

    def __init__(
            self, label_data_path: str, gene_data_dir_path: str,
            gene_freq_file_path: str = None,
            label_data_id_field_name: str = None, label_data_label_field_name: str = None,
            in_memory: bool = False):
        self.label_data_path = label_data_path
        self.gene_data_dir_path = gene_data_dir_path
        self.gene_freq_file_path = gene_freq_file_path
        if label_data_id_field_name:
            self.label_data_id_field_name = label_data_id_field_name
        if label_data_label_field_name:
            self.label_data_label_field_name = label_data_label_field_name

        self.label_df, self.gene_data_file_names = intersection_participants_data(
            label_data_path, gene_data_dir_path=gene_data_dir_path,
            label_data_id_field_name=self.label_data_id_field_name,
        )
        self.label_df = self.data_balance(self.label_df)
        self.gene_data_file_paths_dict = {
            get_gene_file_participant_id(file_name): os.path.join(self.gene_data_dir_path, file_name)
            for file_name in self.gene_data_file_names}
        self.gene_freq = json.load(open(self.gene_freq_file_path, 'r')) if self.gene_freq_file_path else None
        self.label_df.sample()
        self.in_memory = in_memory
        if self.in_memory:
            self.gene_data_in_memory = {label_participant_id: self.get_gene_data(label_participant_id)
                                        for label_participant_id in self.gene_data_file_paths_dict.keys()}

    def get_gene_data(self, label_participant_id):
        gene_file_path = self.gene_data_file_paths_dict[label_participant_id]
        return handle_gene_file(gene_file_path, self.gene_freq)

    def data_balance(self, df: pd.DataFrame):
        label_num = {}
        for label, group_df in df.groupby(self.label_data_label_field_name):
            label_num[label] = group_df.shape[0]
        min_num = min(label_num.values())
        dfs = [group_df.sample(n=min_num)
               for label, group_df in df.groupby(self.label_data_label_field_name)]
        return pd.concat(dfs, axis=0, ignore_index=True)

    def __getitem__(self, index):
        label_participant_id = self.label_df.loc[index, self.label_data_id_field_name]
        if self.in_memory:
            gene_data = self.gene_data_in_memory[label_participant_id]
        else:
            gene_data = self.get_gene_data(label_participant_id)
        label_data = self.label_df.loc[index, self.label_data_label_field_name]
        return (gene_data,), torch.tensor([label_data], dtype=torch.float32)

    def __len__(self):
        return self.label_df.shape[0]


class BertSNPDataset(Dataset):
    label_data_id_field_name = '学籍号'
    label_data_label_field_name = 'high_myopia'

    def __init__(
            self, label_data_path: str, gene_data_dir_path: str, snp_number: int, tokenizer: BertTokenizer,
            label_data_id_field_name: str = None, label_data_label_field_name: str = None,
            in_memory: bool = False):
        self.label_data_path = label_data_path
        self.gene_data_dir_path = gene_data_dir_path
        self.snp_number = snp_number
        self.tokenizer = tokenizer
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

        self.in_memory = in_memory
        if self.in_memory:
            self.gene_data_in_memory = {label_participant_id: self.get_gene_data(label_participant_id)
                                        for label_participant_id in self.gene_data_file_paths_dict.keys()}

    def get_gene_data(self, label_participant_id):
        """
        encode仅返回 input_ids
        encode_plus返回所有编码信息
            input_ids：是单词在词典中的编码
            token_type_ids：区分两个句子的编码（上句全为0，下句全为1）
            attention_mask：指定 对哪些词 进行self-Attention操作
        convert_ids_to_tokens: 将input_ids转化回token
        :param label_participant_id:
        :return:
        """
        gene_file_path = self.gene_data_file_paths_dict[label_participant_id]
        gene_sent = handle_gene_file_bert(gene_file_path)
        encoded_sent = self.tokenizer.encode(gene_sent, add_special_tokens=True)
        input_ids = pad_sequences([encoded_sent], maxlen=self.snp_number + 2, dtype="long",
                                  value=0, truncating="post", padding="post")
        attention_mask = [int(token_id > 0) for token_id in input_ids[0]]
        return torch.tensor(input_ids[0]), torch.tensor(attention_mask)

    def __getitem__(self, index):
        label_participant_id = self.label_df.loc[index, self.label_data_id_field_name]
        if self.in_memory:
            input_id, attention_mask = self.gene_data_in_memory[label_participant_id]
        else:
            input_id, attention_mask = self.get_gene_data(label_participant_id)
        label_data = self.label_df.loc[index, self.label_data_label_field_name]
        # label_data=torch.tensor([label_data], dtype=torch.float32)
        label_data = torch.tensor([0, 1], dtype=torch.float32) if label_data == 1 else torch.tensor([1, 0],
                                                                                                    dtype=torch.float32)
        return (input_id, attention_mask), label_data

    def __len__(self):
        return self.label_df.shape[0]


class ImageDataset(Dataset):
    label_data_id_field_name = '学籍号'
    label_data_label_field_name = 'high_myopia'

    def __init__(
            self, label_data_path: str, image_data_dir_path: str,
            transform=None,
            label_data_id_field_name: str = None, label_data_label_field_name: str = None,
            in_memory: bool = False):
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
        self.label_data_dict = self.set_label_data_dict()

        self.in_memory = in_memory
        if self.in_memory:
            self.image_data_in_memory = [self.get_image_label_data(image_data_file_name)
                                         for image_data_file_name in self.image_data_file_names]

    def set_label_data_dict(self):
        label_data_dict = {}
        for index, row in self.label_df.iterrows():
            if (label_data := row[f'OS_{self.label_data_label_field_name}']) != -1:
                label_data_dict[f"{row[self.label_data_id_field_name]}_OS"] = label_data
            if (label_data := row[f'OD_{self.label_data_label_field_name}']) != -1:
                label_data_dict[f"{row[self.label_data_id_field_name]}_OD"] = label_data
        return label_data_dict

    def get_image_label_data(self, image_file_name):
        label_participant_id, eye_side = image_file_name.split('.')[0].split('_')[0:2]
        label_data = self.label_data_dict[f'{label_participant_id}_{eye_side.upper()}']
        image = Image.open(os.path.join(self.image_data_dir_path, image_file_name))
        # 图片的归一化和标准化应该在transforms中完成
        image = self.transform(image)
        return label_participant_id, image, label_data

    def __getitem__(self, index):
        if self.in_memory:
            label_participant_id, image, label_data = self.image_data_in_memory[index]
        else:
            label_participant_id, image, label_data = self.get_image_label_data(self.image_data_file_names[index])
        return (image,), torch.tensor([label_data], dtype=torch.float32)

    def __len__(self):
        return len(self.image_data_file_names)


class SNPImageDataset(SNPDataset, ImageDataset):
    def __init__(
            self, label_data_path: str, gene_data_dir_path: str, image_data_dir_path: str,
            gene_freq_file_path: str = None, transform: Compose = None,
            label_data_id_field_name: str = None, label_data_label_field_name: str = None,
            in_memory: bool = False):
        self.label_data_path = label_data_path
        self.gene_data_dir_path = gene_data_dir_path
        self.image_data_dir_path = image_data_dir_path
        self.gene_freq_file_path = gene_freq_file_path
        self.transform = transform if transform else base_image_transforms
        if label_data_id_field_name:
            self.label_data_id_field_name = label_data_id_field_name
        if label_data_label_field_name:
            self.label_data_label_field_name = label_data_label_field_name
        self.label_df, self.gene_data_file_names, self.image_data_file_names = intersection_participants_data(
            label_data_path, gene_data_dir_path=gene_data_dir_path, image_data_dir_path=image_data_dir_path,
            label_data_id_field_name=self.label_data_id_field_name,
        )
        self.gene_data_file_paths_dict = {
            get_gene_file_participant_id(file_name): os.path.join(self.gene_data_dir_path, file_name)
            for file_name in self.gene_data_file_names}
        self.gene_freq = json.load(open(self.gene_freq_file_path, 'r')) if self.gene_freq_file_path else None

        self.label_data_dict = self.set_label_data_dict()

        self.in_memory = in_memory
        if self.in_memory:
            self.gene_data_in_memory = {label_participant_id: self.get_gene_data(label_participant_id)
                                        for label_participant_id in self.gene_data_file_paths_dict.keys()}
            self.image_data_in_memory = [self.get_image_label_data(image_data_file_name)
                                         for image_data_file_name in self.image_data_file_names]

    def __getitem__(self, index):
        if self.in_memory:
            label_participant_id, image, label_data = self.image_data_in_memory[index]
            gene_data = self.gene_data_in_memory[label_participant_id]
        else:
            label_participant_id, image, label_data = self.get_image_label_data(self.image_data_file_names[index])
            gene_data = self.get_gene_data(label_participant_id)
        return (gene_data, image), torch.tensor([label_data], dtype=torch.float32)

    def __len__(self):
        return ImageDataset.__len__(self)
