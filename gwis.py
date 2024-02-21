# -*- encoding: utf-8 -*-
"""
@File Name      :   gwis.py
@Create Time    :   2023/9/8 19:35
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

import copy
import json
import os
import random

import click
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from base import work_dirs_dir, checkpoints_dir_name, logs_dir_name
from utils import setup_seed
from utils.compute.metrics import count_metrics_binary_classification
from utils.dir import mk_dir
from utils.early_stopping import LossEarlyStopping
from utils.task import binary_classification_task
from utils.time import datetime_now_str

seed = 2023
lr = 1e-4
step_size = 10
gamma = 0.1
last_epoch = -1
use_early_stopping = True
early_stopping_step = 5
early_stopping_delta = 0

epochs = 100
batch_size = 32
log_interval = 1
save_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子
setup_seed(seed)


class GeneNet(nn.Module):
    def __init__(self, snp_number):
        super().__init__()
        self.snp_number = snp_number
        self.gene_mlp = nn.Sequential(
            nn.Linear(snp_number, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, snps):
        x = self.gene_mlp(snps)
        return x


@click.command()
@click.argument('gene_regions_info_file_path', type=click.Path(exists=True))
@click.argument('data_dir_path', type=click.Path(exists=True))
@click.argument('label_file_path', type=click.Path(exists=True))
@click.option('--label_data_id_field_name', type=str, default='学籍号')
@click.option('--label_data_label_field_name', type=str, default='high_myopia')
@click.option('--train_dir_prefix', type=str, default='')
@click.option('--output_dir_path', type=click.Path(), default=os.getcwd())
def main(
        gene_regions_info_file_path, data_dir_path, label_file_path,
        label_data_id_field_name, label_data_label_field_name, train_dir_prefix,
        output_dir_path
):
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    gene_region_file_names = [file_name for file_name in os.listdir(data_dir_path) if file_name.endswith('.csv')]
    gene_participant_ids = pd.read_csv(os.path.join(data_dir_path, gene_region_file_names[0]), dtype=str)[
        'participant_id'].tolist()
    label_df = pd.read_csv(label_file_path, dtype={label_data_id_field_name: str})
    label_participant_ids = label_df[label_data_id_field_name].tolist()
    participant_ids = list(set(gene_participant_ids) & set(label_participant_ids))
    print(f'ids len:{len(participant_ids)}')
    label_df = label_df[label_df[label_data_id_field_name].isin(participant_ids)]

    train_valid_participant_ids, test_participant_ids = train_test_split(participant_ids, random_state=2023,
                                                                         test_size=0.2)
    test_participant_ids_len = len(test_participant_ids)
    train_participant_ids, valid_participant_ids = train_test_split(train_valid_participant_ids, random_state=2023,
                                                                    test_size=0.2)
    all_labels = {row[label_data_id_field_name]: row[label_data_label_field_name] for index, row in label_df.iterrows()}
    train_labels = torch.tensor([[int(all_labels[participant_id])] for participant_id in train_participant_ids],
                                dtype=torch.float)
    validation_labels = torch.tensor([[int(all_labels[participant_id])] for participant_id in valid_participant_ids],
                                     dtype=torch.float)
    test_labels = torch.tensor([[int(all_labels[participant_id])] for participant_id in test_participant_ids],
                               dtype=torch.float)

    with open(gene_regions_info_file_path, 'r') as f:
        gene_regions_info = json.load(f)

    # 初始化和加载数据
    # todo:后续需要优化，可以逐个生成，而不是一次性全生成出来
    nets = {gene_name: GeneNet(length) for gene_name, length in gene_regions_info['gene_region_snps_len'].items()}
    optimizers = {
        gene_name: optim.Adam(filter(lambda parameter: parameter.requires_grad, nets[gene_name].parameters()), lr=lr)
        for gene_name in gene_regions_info['gene_names']}
    schedulers = {
        gene_name: optim.lr_scheduler.StepLR(optimizers[gene_name], step_size=step_size, gamma=gamma,
                                             last_epoch=last_epoch)
        for gene_name in gene_regions_info['gene_names']}
    criterion = nn.BCEWithLogitsLoss()
    loss_early_stoppings = {
        gene_name: LossEarlyStopping(patience=early_stopping_step, delta=early_stopping_delta, silent=True)
        for gene_name in gene_regions_info['gene_names']}

    gwis_records_dir_name = 'gwis'
    if not train_dir_prefix:
        train_dir_prefix = datetime_now_str()
        print(f'当前时间：{train_dir_prefix}')
        records_dir = os.path.join(work_dirs_dir, gwis_records_dir_name, train_dir_prefix)
        mk_dir(records_dir)

        def train_model(gene_name, length):
            net = nets[gene_name]
            net.to(device)
            optimizer = optimizers[gene_name]
            scheduler = schedulers[gene_name]
            loss_early_stopping = loss_early_stoppings[gene_name]
            # 创建记录文件夹
            checkpoints_dir = os.path.join(records_dir, checkpoints_dir_name, gene_name)
            mk_dir(checkpoints_dir)
            best_model_checkpoint_path = os.path.join(checkpoints_dir, 'best_model_checkpoint.pth')
            writer = SummaryWriter(log_dir=os.path.join(records_dir, logs_dir_name, gene_name))
            writer.add_graph(net, torch.randn(1, length).to(device))
            # 读取数据
            gene_df = pd.read_csv(os.path.join(data_dir_path, f'{gene_name}.csv'), dtype=str)
            gene_df = gene_df[gene_df['participant_id'].isin(train_valid_participant_ids)]
            all_inputs = {participant_id: group_df['val'].astype(int).tolist() for participant_id, group_df in
                          gene_df.groupby('participant_id')}
            train_inputs = torch.tensor([all_inputs[participant_id] for participant_id in train_participant_ids],
                                        dtype=torch.float)
            validation_inputs = torch.tensor([all_inputs[participant_id] for participant_id in valid_participant_ids],
                                             dtype=torch.float)
            # Create the DataLoader for our training set.
            train_data = TensorDataset(train_inputs, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
            # Create the DataLoader for our validation set.
            validation_data = TensorDataset(validation_inputs, validation_labels)
            validation_sampler = SequentialSampler(validation_data)
            validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
            data_loader = {'train': train_dataloader, 'valid': validation_dataloader}

            best_f1 = 0
            for epoch in range(epochs):
                for phase in ['train', 'valid']:
                    if phase == 'train':
                        # 训练
                        net.train()
                    else:
                        # 验证
                        net.eval()
                    y_true, y_pred, y_score = [], [], []
                    running_loss = 0.0
                    for inputs, labels in data_loader[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        y_true += labels.int().reshape(-1).tolist()
                        # 梯度清零
                        optimizer.zero_grad()
                        # 前向传播
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = net(inputs)
                        loss, y_pred_batch, y_score_batch = binary_classification_task(outputs, labels,
                                                                                       criterion=criterion)
                        y_pred += y_pred_batch
                        y_score += y_score_batch
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        running_loss += loss.item()
                    # 计算损失
                    epoch_loss = running_loss / len(data_loader[phase].dataset)
                    # 计算指标
                    all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
                    # 得到最好的模型，需要自己定义哪种情况下指标最好
                    if phase == 'valid' and all_metrics['f1'] > best_f1:
                        best_f1 = all_metrics['f1']
                        state = {
                            'epoch': epoch,
                            'model': net.state_dict(),
                            'best_f1': best_f1,
                            'optimizer': optimizer.state_dict()
                        }
                        torch.save(state, best_model_checkpoint_path)
                    # 记录
                    if epoch % log_interval == 0:
                        writer.add_scalars('Loss', {
                            phase: epoch_loss,
                        }, epoch)
                        writer.add_scalars('acc', {
                            phase: all_metrics['acc'],
                        }, epoch)
                        writer.add_scalars('mcc', {
                            phase: all_metrics['mcc'],
                        }, epoch)
                        writer.add_scalars('precision', {
                            phase: all_metrics['precision'],
                        }, epoch)
                        writer.add_scalars('recall', {
                            phase: all_metrics['recall'],
                        })
                        writer.add_scalars('f1', {
                            phase: all_metrics['f1'],
                        })
                        writer.add_scalars('tpr', {
                            phase: all_metrics['tpr'],
                        })
                        writer.add_scalars('fpr', {
                            phase: all_metrics['fpr'],
                        })
                        writer.add_scalars('ks', {
                            phase: all_metrics['ks'],
                        }, epoch)
                        writer.add_scalars('sp', {
                            phase: all_metrics['sp'],
                        }, epoch)
                        writer.add_scalars('auc', {
                            phase: all_metrics['auc_score'],
                        }, epoch)
                        writer.flush()
                        # 判断是否早停
                    if use_early_stopping and phase == 'valid':
                        loss_early_stopping(epoch_loss)
                scheduler.step()
                if epoch % step_size == 0:
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                if epoch % save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'best_f1': best_f1,
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(checkpoints_dir, f'epoch_{epoch}_model_checkpoints.pth'))
                if use_early_stopping and loss_early_stopping.early_stop:
                    break
            writer.close()

        for gene_name, length in tqdm.tqdm(gene_regions_info['gene_region_snps_len'].items()):
            train_model(gene_name, length)

        # from concurrent.futures import ThreadPoolExecutor, wait
        # def list_to_n_group(list_to_group: list, n: int = 3) -> list:
        #     length = len(list_to_group)
        #     remainder = length % n
        #     if remainder == 0:
        #         step = length // n
        #     else:
        #         step = length // n + 1
        #     return [list_to_group[i:i + step] for i in range(0, len(list_to_group), step)]
        #
        #
        # for group in list_to_n_group(gene_regions_info['gene_region_snps_len'].items(), 16):
        #     with ThreadPoolExecutor() as executor:
        #         tasks = [executor.submit(train_model, gene_name, length) for gene_name, length in group]
        #         wait(tasks)
    else:
        records_dir = os.path.join(work_dirs_dir, gwis_records_dir_name, train_dir_prefix)
        # random.seed()
        # 计算group-wise importance score
        k_random_times = 1000
        # 记录所有的随机损失，非常占用内存
        # fake_losses = []
        delta_losses = []
        delta_losses2 = []
        all_genes = []
        for gene_name in tqdm.tqdm(gene_regions_info['gene_names']):
            net = nets[gene_name]
            net.to(device)
            checkpoints_dir = os.path.join(records_dir, checkpoints_dir_name, gene_name)
            best_model_checkpoint_path = os.path.join(checkpoints_dir, 'best_model_checkpoint.pth')
            if not os.path.exists(best_model_checkpoint_path):
                print(f'{gene_name} best_model_checkpoint_path not exists')
                continue
            net.load_state_dict(torch.load(best_model_checkpoint_path)['model'], strict=False)
            test_criterion = nn.BCEWithLogitsLoss(reduction='none')

            gene_df = pd.read_csv(os.path.join(data_dir_path, f'{gene_name}.csv'), dtype=str)
            gene_df = gene_df[gene_df['participant_id'].isin(test_participant_ids)]
            all_inputs = {participant_id: group_df['val'].astype(int).tolist() for participant_id, group_df in
                          gene_df.groupby('participant_id')}
            test_inputs = torch.tensor([all_inputs[participant_id] for participant_id in test_participant_ids],
                                       dtype=torch.float)

            test_data = TensorDataset(test_inputs, test_labels)
            test_sampler = RandomSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

            y_true, y_outputs, true_losses = [], [], []
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_true += labels.int().reshape(-1).tolist()
                with torch.set_grad_enabled(False):
                    outputs = net(inputs)
                new_outputs = torch.sigmoid(outputs)
                y_outputs += new_outputs.reshape(-1).tolist()
                loss = test_criterion(new_outputs, labels.float()).clone().tolist()
                true_losses += loss
            true_losses = [i[0] for i in true_losses]

            e_fake_losses = [[] for i in range(test_participant_ids_len)]
            for k in range(k_random_times):
                y_outputs_copy = copy.deepcopy(y_outputs)
                random.shuffle(y_outputs_copy)
                # random_losses = [
                #     test_criterion(torch.tensor(outputs, dtype=torch.float),
                #                    torch.tensor(labels, dtype=torch.float)).item()
                #     for outputs, labels in zip(y_outputs_copy, y_true)]
                random_losses = test_criterion(torch.tensor(y_outputs_copy, dtype=torch.float),
                                               torch.tensor(y_true, dtype=torch.float)).clone().tolist()
                for i in range(test_participant_ids_len):
                    e_fake_losses[i].append(random_losses[i])
            # fake_losses.append(e_fake_losses)
            e_fake_loss = [sum(e_fake_losses[i]) / k_random_times for i in range(test_participant_ids_len)]
            delta_loss = sum([true_losses[i] - e_fake_loss[i] for i in range(test_participant_ids_len)])
            delta_losses.append(float(delta_loss))

            # 另一种写法，同时使用会因为随机的数组不同导致最后结果有略微的差异，但是大体不影响，偏差值大概在0.01左右
            true_losses = np.array(true_losses)
            k_fake_losses = []
            for k in range(k_random_times):
                y_outputs_copy = copy.deepcopy(y_outputs)
                random.shuffle(y_outputs_copy)
                random_losses = test_criterion(torch.tensor(y_outputs_copy, dtype=torch.float),
                                               torch.tensor(y_true, dtype=torch.float)).clone().numpy()
                k_fake_losses.append(np.sum(true_losses - random_losses))
            delta_loss2 = sum(k_fake_losses) / k_random_times
            delta_losses2.append(float(delta_loss2))

            all_genes.append(gene_name)
        gwis_file_path = os.path.join(output_dir_path, 'gwis.json')
        with open(gwis_file_path, 'w') as w:
            print(f'group-wise importance score is saved to {gwis_file_path}')
            json.dump({
                # "fake_losses": fake_losses,
                "delta_losses": delta_losses,
                "delta_losses2": delta_losses2,
                "all_genes": all_genes
            }, w)


if __name__ == '__main__':
    """
    训练
    group wise importance score
    python gwis.py work_dirs/data/gene/students_snps_all_frequency_0.001/gene_regions/gene_regions_info.json \
    work_dirs/data/gene/students_snps_all_frequency_0.001/gene_regions/ \
    work_dirs/data/label/all_students_qc_two_eyes_merge_20230919183434.csv
    
    测试
    python gwis.py work_dirs/data/gene/students_snps_all_frequency_0.05/gene_regions/gene_regions_info.json \
    work_dirs/data/gene/students_snps_all_frequency_0.05/gene_regions/ \
    work_dirs/data/label/all_students_qc_two_eyes_merge_20230919183434.csv \
    --train_dir_prefix 20231213144941 \
    --output_dir_path work_dirs/data/gene/students_snps_all_frequency_0.05/label_20230919183434/
    """
    # data_dir_path = '/pub/data/sunhj/Multi-Modality-SNPCorrelateImage/data/gene/students_gene_regions_snps/gene_regions'
    # gene_regions_info_file_path = "/pub/data/sunhj/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf/all_gene_regions_info.json"
    # label_file_path = "/pub/data/sunhj/Multi-Modality-SNPCorrelateImage/data/label/all_students_qc_two_eyes_merge.csv"
    main()
