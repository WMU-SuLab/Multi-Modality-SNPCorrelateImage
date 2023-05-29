# -*- encoding: utf-8 -*-
"""
@File Name      :   make_data_loaders.py
@Create Time    :   2023/4/10 10:07
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

from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from .datasets import SNPDataset, ImageDataset, SNPImageDataset
from .multi_gpus import get_world_size, get_rank
from .transforms import gene_image_transforms


def mk_data_loaders_single(
        dataset,
        train_dataset_args: tuple, train_dataset_kwargs: dict,
        valid_dataset_args: tuple, valid_dataset_kwargs: dict,
        test_dataset_judge_args: tuple, test_dataset_args: tuple, test_dataset_kwargs: dict,
        batch_size: int = 8, drop_last: bool = False):
    data_loaders = {
        'train': DataLoader(
            dataset(*train_dataset_args, **train_dataset_kwargs),
            batch_size=batch_size, shuffle=True, drop_last=drop_last,
            num_workers=4, pin_memory=True,
        ),
        'valid': DataLoader(
            dataset(*valid_dataset_args, **valid_dataset_kwargs),
            batch_size=batch_size, shuffle=True, drop_last=drop_last,
            num_workers=4, pin_memory=True,
        ),
    }
    if test_dataset_judge_args and all(
            [os.listdir(test_dataset_judge_arg) for test_dataset_judge_arg in test_dataset_judge_args]):
        data_loaders['test'] = DataLoader(
            dataset(*test_dataset_args, **test_dataset_kwargs),
            batch_size=batch_size, shuffle=True, drop_last=drop_last,
            num_workers=4, pin_memory=True,
        )
    return data_loaders


def mk_data_loaders_multi(
        dataset,
        train_dataset_args: tuple, train_dataset_kwargs: dict,
        valid_dataset_args: tuple, valid_dataset_kwargs: dict,
        test_dataset_judge_args: tuple, test_dataset_args: tuple, test_dataset_kwargs: dict,
        batch_size: int = 8, drop_last: bool = False):
    train_dataset = dataset(*train_dataset_args, **train_dataset_kwargs)
    train_distributed_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
    train_batch_sampler = BatchSampler(train_distributed_sampler, batch_size=batch_size, drop_last=drop_last)
    train_data_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        # num_workers=4, pin_memory=True,
    )
    valid_dataset = dataset(*valid_dataset_args, **valid_dataset_kwargs)
    valid_distributed_sampler = DistributedSampler(valid_dataset, num_replicas=get_world_size(), rank=get_rank())
    valid_batch_sampler = BatchSampler(valid_distributed_sampler, batch_size=batch_size, drop_last=drop_last)
    valid_data_loader = DataLoader(
        valid_dataset, batch_sampler=valid_batch_sampler,
        # num_workers=4, pin_memory=True,
    )
    data_loaders = {'train': train_data_loader, 'valid': valid_data_loader}
    samplers = {'train': train_distributed_sampler, 'valid': valid_distributed_sampler}
    if test_dataset_judge_args and all(
            [os.listdir(test_dataset_judge_arg) for test_dataset_judge_arg in test_dataset_judge_args]):
        test_dataset = dataset(*test_dataset_args, **test_dataset_kwargs)
        test_distributed_sampler = DistributedSampler(test_dataset, num_replicas=get_world_size(), rank=get_rank())
        test_batch_sampler = BatchSampler(test_distributed_sampler, batch_size=batch_size, drop_last=drop_last)
        test_data_loader = DataLoader(
            test_dataset, batch_sampler=test_batch_sampler,
            # num_workers=4, pin_memory=True,
        )
        data_loaders['test'] = test_data_loader
        samplers['test'] = test_distributed_sampler
    return data_loaders, samplers


class MkDataLoaders:
    def __init__(self, data_loader_type):
        self.data_loader_type = data_loader_type
        if data_loader_type == 'single':
            self.mk_data_loaders = mk_data_loaders_single
        elif data_loader_type == 'multi':
            self.mk_data_loaders = mk_data_loaders_multi
        else:
            raise ValueError('data_loader_type must be single or multi')


class MkSNPNetDataLoaders(MkDataLoaders):
    def __call__(
            self, data_paths: dict, gene_freq_file_path: str = None,
            batch_size: int = 8, drop_last: bool = False, **kwargs):
        return self.mk_data_loaders(
            SNPDataset,
            # train
            (data_paths['train_label_file_path'], data_paths['train_gene_dir_path'], gene_freq_file_path,),
            kwargs,
            # valid
            (data_paths['valid_label_file_path'], data_paths['valid_gene_dir_path'], gene_freq_file_path,),
            kwargs,
            # test
            (data_paths['test_gene_dir_path'],),
            (data_paths['test_label_file_path'], data_paths['test_gene_dir_path'], gene_freq_file_path,),
            kwargs,
            batch_size=batch_size, drop_last=drop_last
        )


class MkImageNetDataLoaders(MkDataLoaders):
    def __call__(
            self, data_paths: dict, batch_size: int = 8, drop_last: bool = False, **kwargs):
        return self.mk_data_loaders(
            ImageDataset,
            # train
            (data_paths['train_label_file_path'], data_paths['train_image_dir_path'], gene_image_transforms['train']),
            kwargs,
            # valid
            (data_paths['valid_label_file_path'], data_paths['valid_image_dir_path'], gene_image_transforms['valid']),
            kwargs,
            # test
            (data_paths['test_image_dir_path'],),
            (data_paths['test_label_file_path'], data_paths['test_image_dir_path'], gene_image_transforms['test']),
            kwargs,
            batch_size=batch_size, drop_last=drop_last
        )


class MkSNPImageNetDataLoaders(MkDataLoaders):
    def __call__(
            self, data_paths: dict, gene_freq_file_path: str = None,
            batch_size: int = 8, drop_last: bool = False, **kwargs):
        return self.mk_data_loaders(
            SNPImageDataset,
            # train
            (data_paths['train_label_file_path'], data_paths['train_gene_dir_path'], data_paths['train_image_dir_path'],
             gene_freq_file_path, gene_image_transforms['train']),
            kwargs,
            # valid
            (data_paths['valid_label_file_path'], data_paths['valid_gene_dir_path'], data_paths['valid_image_dir_path'],
             gene_freq_file_path, gene_image_transforms['valid']),
            kwargs,
            # test
            (data_paths['test_gene_dir_path'], data_paths['test_image_dir_path']),
            (data_paths['test_label_file_path'], data_paths['test_gene_dir_path'], data_paths['test_image_dir_path'],
             gene_freq_file_path, gene_image_transforms['test']),
            kwargs,
            batch_size=batch_size, drop_last=drop_last
        )


mk_data_loaders_single_funcs = {
    'SNPNet': MkSNPNetDataLoaders('single'),
    'ImageNet': MkImageNetDataLoaders('single'),
    'SNPImageNet': MkSNPImageNetDataLoaders('single'),
}

mk_data_loaders_multi_funcs = {
    'SNPNet': MkSNPNetDataLoaders('multi'),
    'ImageNet': MkImageNetDataLoaders('multi'),
    'SNPImageNet': MkSNPImageNetDataLoaders('multi'),
}

__all__ = ['mk_data_loaders_single_funcs', 'mk_data_loaders_multi_funcs']
