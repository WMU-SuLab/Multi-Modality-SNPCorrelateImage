# -*- encoding: utf-8 -*-
"""
@File Name      :   dataset.py
@Create Time    :   2022/12/6 11:46
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

import click

from base import data_divide_dir
from utils.dir import mk_dir
from utils.name import get_image_file_participant_id, get_gene_file_participant_id
from utils.participants import intersection_participants_ids_with_path
from utils.time import datetime_now


def train_test_ratio(train_ratio: int = 7, test_ratio: int = 3):
    all_ratio = train_ratio + test_ratio
    return train_ratio / all_ratio, test_ratio / all_ratio


def train_valid_ratio(train_ratio: int = 7, valid_ratio: int = 3):
    all_ratio = train_ratio + valid_ratio
    return train_ratio / all_ratio, valid_ratio / all_ratio


def train_valid_test_ratio(train_ratio: int = 5, valid_ratio: int = 1, test_ratio: int = 1):
    all_ratio = train_ratio + valid_ratio + test_ratio
    return train_ratio / all_ratio, valid_ratio / all_ratio, test_ratio / all_ratio


def mk_dataset_root_dir_path(dataset_divide_dir: str = None) -> str:
    if not dataset_divide_dir:
        dataset_divide_dir = data_divide_dir
    dataset_dir_path = os.path.join(dataset_divide_dir, datetime_now())
    mk_dir(dataset_dir_path)
    return dataset_dir_path


def generate_label_file_path(dir_path: str) -> str:
    return os.path.join(dir_path, 'labels.csv')


def mk_dataset_paths(dataset_dir_path: str):
    paths = {'dataset_dir_path': dataset_dir_path}
    # 数据集目录划分
    train_dir_path = os.path.join(dataset_dir_path, 'train')
    valid_dir_path = os.path.join(dataset_dir_path, 'valid')
    test_dir_path = os.path.join(dataset_dir_path, 'test')
    mk_dir(train_dir_path)
    mk_dir(valid_dir_path)
    mk_dir(test_dir_path)
    paths['train_dir_path'] = train_dir_path
    paths['valid_dir_path'] = valid_dir_path
    paths['test_dir_path'] = test_dir_path
    # label标签文件路径
    train_label_file_path = generate_label_file_path(train_dir_path)
    valid_label_file_path = generate_label_file_path(valid_dir_path)
    test_label_file_path = generate_label_file_path(test_dir_path)
    paths['train_label_file_path'] = train_label_file_path
    paths['valid_label_file_path'] = valid_label_file_path
    paths['test_label_file_path'] = test_label_file_path
    # gene数据文件夹
    train_gene_dir_path = os.path.join(train_dir_path, 'gene')
    valid_gene_dir_path = os.path.join(valid_dir_path, 'gene')
    test_gene_dir_path = os.path.join(test_dir_path, 'gene')
    mk_dir(train_gene_dir_path)
    mk_dir(valid_gene_dir_path)
    mk_dir(test_gene_dir_path)
    paths['train_gene_dir_path'] = train_gene_dir_path
    paths['valid_gene_dir_path'] = valid_gene_dir_path
    paths['test_gene_dir_path'] = test_gene_dir_path
    # image数据文件夹
    train_image_dir_path = os.path.join(train_dir_path, 'image')
    valid_image_dir_path = os.path.join(valid_dir_path, 'image')
    test_image_dir_path = os.path.join(test_dir_path, 'image')
    mk_dir(train_image_dir_path)
    mk_dir(valid_image_dir_path)
    mk_dir(test_image_dir_path)
    paths['train_image_dir_path'] = train_image_dir_path
    paths['valid_image_dir_path'] = valid_image_dir_path
    paths['test_image_dir_path'] = test_image_dir_path
    return paths


def gene_data_mk_link(participant_ids: list, gene_data_dir_path: str, new_gene_data_dir_path: str):
    gene_data_file_names = [
        gene_data_file_name
        for gene_data_file_name in os.listdir(gene_data_dir_path)
        if get_gene_file_participant_id(gene_data_file_name) in participant_ids
    ]
    for gene_data_file_name in gene_data_file_names:
        os.symlink(
            os.path.join(gene_data_dir_path, gene_data_file_name),
            os.path.join(new_gene_data_dir_path, gene_data_file_name)
        )


def image_data_mk_link(participant_ids: list, image_data_dir_path: str, new_image_data_dir_path: str):
    image_data_file_names = [
        image_data_file_name
        for image_data_file_name in os.listdir(image_data_dir_path)
        if get_image_file_participant_id(image_data_file_name) in participant_ids
    ]
    for image_data_file_name in image_data_file_names:
        os.symlink(
            os.path.join(image_data_dir_path, image_data_file_name),
            os.path.join(new_image_data_dir_path, image_data_file_name)
        )


# 本模块的划分方法只适用于这个数据集，但是其他数据集可以借鉴这种思路
@click.command()
@click.argument('label_data_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--gene_data_dir_path', type=click.Path(exists=True, file_okay=False), default=None)
@click.option('--image_data_dir_path', type=click.Path(exists=True, file_okay=False), default=None)
@click.option('--dataset_divide_dir', type=click.Path(exists=True, file_okay=False), default=None)
@click.option('--label_data_id_field_name', type=str, default='Participant ID')
@click.option('--train_ratio', type=int, default=7)
@click.option('--valid_ratio', type=int, default=3)
@click.option('--test_ratio', type=int, default=0)
@click.option('--strategy', type=click.Choice(['combination', 'separateness', 'train_valid']), default='train_valid')
def train_valid_test_split(
        label_data_path: str, gene_data_dir_path: str, image_data_dir_path: str,
        dataset_divide_dir: str, label_data_id_field_name: str,
        train_ratio, valid_ratio, test_ratio, strategy: str,
):
    """
    :param label_data_path:
    :param gene_data_dir_path:
    :param image_data_dir_path:
    :param dataset_divide_dir:
    :param label_data_id_field_name: 需要注意修改id列的列名
    :param train_ratio:
    :param valid_ratio:
    :param test_ratio:
    :param strategy:
    :return:
    """
    if strategy == 'combination':
        train_rate, valid_rate, test_rate = train_valid_test_ratio(train_ratio, valid_ratio, test_ratio)
    elif strategy == 'separateness':
        train_valid_rate, test_rate = train_test_ratio(train_ratio, test_ratio)
        train_rate, valid_rate = train_valid_ratio(train_ratio, valid_ratio)
        train_rate, valid_rate = train_rate * train_valid_rate, valid_rate * train_valid_rate
    elif strategy == 'train_valid':
        train_rate, valid_rate = train_valid_ratio(train_ratio, valid_ratio)
    else:
        raise ValueError('strategy error')
    dataset_root_dir_path = mk_dataset_root_dir_path(dataset_divide_dir)
    data_paths = mk_dataset_paths(dataset_root_dir_path)
    # label data
    # attention: 这里的 dtype 是为了防止 id 被转换成科学计数法
    participants_ids, label_df, gene_file_names, image_file_names = intersection_participants_ids_with_path(
        label_data_path, gene_data_dir_path, image_data_dir_path, label_data_id_field_name
    )
    label_df = label_df[label_df[label_data_id_field_name].isin(participants_ids)]
    label_participants = label_df[label_data_id_field_name].tolist()
    label_df_length = len(label_participants)
    print(f'筛选完后可用的: {label_df_length}')
    # 打乱
    label_df = label_df.sample(frac=1).reset_index(drop=True)
    # 划分
    train_label_df = label_df[:round(label_df_length * train_rate)]
    valid_label_df = label_df[round(label_df_length * train_rate):round(label_df_length * (train_rate + valid_rate))]
    test_label_df = label_df[round(label_df_length * (train_rate + valid_rate)):]
    train_label_df.to_csv(data_paths['train_label_file_path'], index=False)
    valid_label_df.to_csv(data_paths['valid_label_file_path'], index=False)
    test_label_df.to_csv(data_paths['test_label_file_path'], index=False)
    # 获取id
    train_participant_ids = train_label_df[label_data_id_field_name].drop_duplicates().astype(str).tolist()
    valid_participant_ids = valid_label_df[label_data_id_field_name].drop_duplicates().astype(str).tolist()
    test_participant_ids = test_label_df[label_data_id_field_name].drop_duplicates().astype(str).tolist()
    # gene data
    if gene_data_dir_path:
        gene_data_mk_link(train_participant_ids, gene_data_dir_path, data_paths['train_gene_dir_path'])
        gene_data_mk_link(valid_participant_ids, gene_data_dir_path, data_paths['valid_gene_dir_path'])
        gene_data_mk_link(test_participant_ids, gene_data_dir_path, data_paths['test_gene_dir_path'])
    # image data
    if image_data_dir_path:
        image_data_mk_link(train_participant_ids, image_data_dir_path, data_paths['train_image_dir_path'])
        image_data_mk_link(valid_participant_ids, image_data_dir_path, data_paths['valid_image_dir_path'])
        image_data_mk_link(test_participant_ids, image_data_dir_path, data_paths['test_image_dir_path'])
    with open(os.path.join(dataset_root_dir_path, 'divide_info.txt'), 'w') as f:
        f.write(f'label data path: {label_data_path}\n')
        f.write(f'gene data dir path: {gene_data_dir_path}\n')
        f.write(f'image data dir path: {image_data_dir_path}\n')
        f.write(f'strategy: {strategy}\n')
        f.write(f'train ratio: {train_ratio}\n')
        f.write(f'valid ratio: {valid_ratio}\n')
        f.write(f'test ratio: {test_ratio}\n')
        f.write(f'train: {len(train_participant_ids)}\n')
        f.write(f'valid: {len(valid_participant_ids)}\n')
        f.write(f'test: {len(test_participant_ids)}\n')


if __name__ == '__main__':
    train_valid_test_split()
