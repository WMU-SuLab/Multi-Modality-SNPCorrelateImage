# -*- encoding: utf-8 -*-
"""
@File Name      :   available_participants.py
@Create Time    :   2023/3/18 11:23
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

import pandas as pd

from .text_handler.name import get_label_participant_id, get_gene_file_participant_id, \
    get_image_file_participant_id_instance, get_image_file_participant_id


def intersection_participant_ids(
        label_ids: list, gene_file_names: list = None, image_file_names=None):
    # 标签的id不能为空，图片或者基因如果为空则不进行考虑
    gene_ids = [get_gene_file_participant_id(file_name) for file_name in gene_file_names] if gene_file_names else []
    image_ids = [get_image_file_participant_id(file_name) for file_name in image_file_names] if image_file_names else []
    label_ids = [get_label_participant_id(label_item) for label_item in label_ids]
    ids = set(label_ids)
    if gene_ids:
        ids = set(gene_ids) & ids
    if image_ids:
        ids = set(image_ids) & ids
    return list(ids)


def intersection_participants_ids_with_path(
        label_data_path: str, gene_data_dir_path: str = None, image_data_dir_path: str = None,
        label_data_id_field_name='学籍号'):
    label_df = pd.read_csv(label_data_path, dtype={
        label_data_id_field_name: str,
    })
    label_df[label_data_id_field_name] = label_df[label_data_id_field_name].str.split('_').str[0]
    label_df[label_data_id_field_name] = label_df[label_data_id_field_name].sort_values()
    label_ids = label_df[label_data_id_field_name].tolist()
    gene_file_names = [
        gene_file_name
        for gene_file_name in os.listdir(gene_data_dir_path)
        # if gene_file_name.endswith('.csv')
    ] if gene_data_dir_path else []
    image_file_names = [
        image_file_name
        for image_file_name in os.listdir(image_data_dir_path)
        # if image_file_name.endswith('.png')
    ] if image_data_dir_path else []
    ids = intersection_participant_ids(label_ids, gene_file_names, image_file_names)
    return ids, label_df, gene_file_names, image_file_names


def intersection_participants_data(
        label_data_path: str, gene_data_dir_path: str = None, image_data_dir_path: str = None,
        label_data_id_field_name='学籍号'):
    ids, label_df, gene_file_names, image_file_names = intersection_participants_ids_with_path(
        label_data_path, gene_data_dir_path, image_data_dir_path, label_data_id_field_name
    )
    label_df = label_df[label_df[label_data_id_field_name].apply(get_label_participant_id).isin(ids)]
    label_df_ids = label_df[label_data_id_field_name].tolist()
    gene_data_file_names = [
        file_name
        for file_name in gene_file_names
        if get_gene_file_participant_id(file_name) in ids]
    image_data_file_names = [
        file_name
        for file_name in image_file_names
        if get_image_file_participant_id_instance(file_name) in label_df_ids]
    data = [label_df]
    if gene_data_dir_path:
        data.append(gene_data_file_names)
    if image_data_dir_path:
        data.append(image_data_file_names)
    return tuple(data)
