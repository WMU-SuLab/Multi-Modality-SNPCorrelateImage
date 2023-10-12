# -*- encoding: utf-8 -*-
"""
@File Name      :   gene_group_participants.py
@Create Time    :   2023/9/8 13:08
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

import json
import multiprocessing
import os
import time
import random
import copy
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from base import work_dirs_dir
from utils import setup_seed
from utils.compute.metrics import count_metrics_binary_classification
from utils.dir import mk_dir
from utils.early_stopping import LossEarlyStopping
from utils.lists import list_to_n_group
from utils.task import binary_classification_task
from utils.time import datetime_now_str

data_dir_path = r'F:\sunhj\Multi-Modality-SNPCorrelateImage\data\gene\students_gene_regions_snps\participants'
gene_regions_snps_info_file_path = r"F:\sunhj\Multi-Modality-SNPCorrelateImage\data\gene\filtered_alleles_vcf\all_gene_regions_info.json"
label_file_path = r"F:\sunhj\Multi-Modality-SNPCorrelateImage\data\label\all_students_qc_two_eyes_merge.csv"
label_data_id_field_name = '学籍号'
label_data_label_field_name = 'high_myopia'

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
            nn.Linear(snp_number, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, snps):
        x = self.gene_mlp(snps)
        return x


gene_participant_ids = [dir_name for dir_name in os.listdir(data_dir_path) if
                        not dir_name.endswith('.csv')]
label_df = pd.read_csv(label_file_path, dtype={label_data_id_field_name: str})
label_participant_ids = label_df[label_data_id_field_name].tolist()
participant_ids = list(set(gene_participant_ids) & set(label_participant_ids))
label_df = label_df[label_df[label_data_id_field_name].isin(participant_ids)]

participants_dir_path = [dir_path for participant_id in participant_ids
                         if os.path.isdir(dir_path := os.path.join(data_dir_path, participant_id))]
label_dict = {}
for index, row in label_df.iterrows():
    label_dict[row[label_data_id_field_name]] = row[label_data_label_field_name]
train_valid_participants_dir_path, test_participants_dir_path = train_test_split(participants_dir_path,
                                                                                 random_state=2023, test_size=0.2)
with open(gene_regions_snps_info_file_path, 'r') as f:
    gene_regions_snps_info = json.load(f)
# 初始化和加载数据
nets = {gene_name: GeneNet(length) for gene_name, length in gene_regions_snps_info['gene_regions_len'].items()}
optimizers = {
    gene_name: optim.Adam(filter(lambda parameter: parameter.requires_grad, nets[gene_name].parameters()), lr=lr)
    for gene_name in gene_regions_snps_info['gene_regions']}
schedulers = {
    gene_name: optim.lr_scheduler.StepLR(optimizers[gene_name], step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    for gene_name in gene_regions_snps_info['gene_regions']}
criterion = nn.BCEWithLogitsLoss()
loss_early_stoppings = {
    gene_name: LossEarlyStopping(patience=early_stopping_step, delta=early_stopping_delta, silent=True)
    for gene_name in gene_regions_snps_info['gene_regions']}


def read_csv(csv_file_path):
    csv_df = pd.read_csv(csv_file_path)
    return csv_df['val'].tolist()


# 训练模型
def train_model(gene_regions_snps_group, records_dir):
    for gene_name, length in gene_regions_snps_group:
        net = nets[gene_name]
        net.to(device)
        optimizer = optimizers[gene_name]
        scheduler = schedulers[gene_name]
        loss_early_stopping = loss_early_stoppings[gene_name]

        checkpoints_dir = os.path.join(records_dir, 'checkpoints', gene_name)
        mk_dir(checkpoints_dir)
        best_model_checkpoint_path = os.path.join(checkpoints_dir, 'best_model_checkpoint.pth')
        wts_dir = os.path.join(records_dir, 'wts', gene_name)
        mk_dir(wts_dir)
        best_model_wts_path = os.path.join(wts_dir, 'best_model_wts.pth')
        writer = SummaryWriter(log_dir=os.path.join(records_dir, 'logs', gene_name))
        writer.add_graph(net, torch.randn(1, length).to(device))
        # 读取数据
        read_data_start_time = time.time()
        # 多线程读
        # from concurrent.futures import ThreadPoolExecutor, wait
        # with ThreadPoolExecutor() as executor:
        #     tasks = [executor.submit(read_csv, os.path.join(participant_dir_path, f'{gene_name}.csv'))
        #              for participant_dir_path in train_valid_participants_dir_path]
        #     wait(tasks)
        #     data_dict = {os.path.basename(participant_dir_path): task.result() for participant_dir_path, task in
        #                  zip(train_valid_participants_dir_path, tasks)}
        data_dict = {}
        for participant_dir_path in train_valid_participants_dir_path:
            participant_id = os.path.basename(participant_dir_path)
            df = pd.read_csv(os.path.join(participant_dir_path, f'{gene_name}.csv'))
            val = df['val'].tolist()
            data_dict[participant_id] = val

        read_data_end_time = time.time()
        print(f"读取数据时间:{read_data_end_time - read_data_start_time}")

        data = []
        data_labels = []
        for participant_id in data_dict.keys():
            data.append(data_dict[participant_id])
            data_labels.append([label_dict[participant_id]])
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(data, data_labels,
                                                                                            random_state=2023,
                                                                                            test_size=0.2)
        train_inputs = torch.tensor(train_inputs, dtype=torch.float)
        validation_inputs = torch.tensor(validation_inputs, dtype=torch.float)
        train_labels = torch.tensor(train_labels, dtype=torch.float)
        validation_labels = torch.tensor(validation_labels, dtype=torch.float)
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
                    loss, y_pred_batch, y_score_batch = binary_classification_task(outputs, labels, criterion=criterion)
                    y_pred += y_pred_batch
                    y_score += y_score_batch
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    running_loss += loss.item()
                # 计算损失
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                # 计算指标
                all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
                acc, mcc, precision, recall, f1, fpr, tpr, ks, sp, auc_score = all_metrics
                # 得到最好的模型，需要自己定义哪种情况下指标最好
                if phase == 'valid' and f1 > best_f1:
                    best_f1 = f1
                    torch.save(net.state_dict(), best_model_wts_path)
                    state = {
                        'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'best_f1': best_f1,
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(state, best_model_checkpoint_path)
                # 记录
                if epoch % log_interval == 0:
                    writer.add_scalars('Loss', {
                        phase: epoch_loss,
                    }, epoch)
                    writer.add_scalars('ACC', {
                        phase: acc,
                    }, epoch)
                    writer.add_scalars('MCC', {
                        phase: mcc,
                    }, epoch)
                    writer.add_scalars('Precision', {
                        phase: precision,
                    }, epoch)
                    writer.add_scalars('Recall', {
                        phase: recall,
                    }, epoch)
                    writer.add_scalars('F1', {
                        phase: f1,
                    }, epoch)
                    writer.add_scalars('FPR', {
                        phase: fpr,
                    }, epoch)
                    writer.add_scalars('TPR', {
                        phase: tpr,
                    }, epoch)
                    writer.add_scalars('KS', {
                        phase: ks,
                    }, epoch)
                    writer.add_scalars('SP', {
                        phase: sp,
                    }, epoch)
                    writer.add_scalars('AUC', {
                        phase: auc_score,
                    }, epoch)
                    writer.flush()
                    # 判断是否早停
                if use_early_stopping and phase == 'valid':
                    loss_early_stopping(epoch_loss)
            if epoch % step_size == 0:
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            if epoch % save_interval == 0:
                torch.save(net.state_dict(), os.path.join(wts_dir, f'epoch_{epoch}_model_wts.pth'))
                torch.save({
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict()
                }, os.path.join(checkpoints_dir, f'epoch_{epoch}_model_checkpoints.pth'))
            if use_early_stopping and loss_early_stopping.early_stop:
                break
        writer.close()


def main():
    train_dir_prefix = datetime_now_str()
    print(f'当前时间：{train_dir_prefix}')
    records_dir = os.path.join(work_dirs_dir, 'gene_records', train_dir_prefix)
    mk_dir(records_dir)

    train_model(gene_regions_snps_info['gene_regions_len'].items(), records_dir)

    # pool = multiprocessing.Pool(processes=4)
    # for gene_regions_snps_group in list_to_n_group(gene_regions_snps_info['gene_regions_len'].items(), 4):
    #     pool.apply_async(train_model, (gene_regions_snps_group, records_dir))
    # pool.close()
    # pool.join()

    # 计算group-wise importance score
    k_random_times = 10
    selected_genes = []
    for gene_name in gene_regions_snps_info['gene_regions']:
        net = nets[gene_name]
        wts_dir = os.path.join(records_dir, 'wts', gene_name)
        best_model_wts_path = os.path.join(wts_dir, 'best_model_wts.pth')
        if not os.path.exists(best_model_wts_path):
            best_model_wts_path = os.path.join(wts_dir, 'epoch_0_model_wts.pth')
        net.load_state_dict(torch.load(best_model_wts_path), strict=False)

        test_data_dict = {}
        for participant_dir_path in test_participants_dir_path:
            participant_id = os.path.basename(participant_dir_path)
            df = pd.read_csv(os.path.join(participant_dir_path, f'{gene_name}.csv'))
            val = df['val'].tolist()
            test_data_dict[participant_id] = val

        test_data = []
        test_data_labels = []
        for participant_id in test_data_dict.keys():
            test_data.append(test_data_dict[participant_id])
            test_data_labels.append(label_dict[participant_id])

        test_inputs = torch.tensor(test_data)
        test_labels = torch.tensor(test_data_labels)
        test_data = TensorDataset(test_inputs, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        y_true, y_outputs, true_losses = [], [], []
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_true += labels.int().reshape(-1).tolist()
            with torch.set_grad_enabled(False):
                outputs = net(*inputs)
            new_outputs = torch.sigmoid(outputs)
            y_outputs.append(new_outputs)
            loss = criterion(new_outputs, labels.float())
            true_losses.append(loss.item())

        e_losses = []
        for i in range(k_random_times):
            y_outputs_copy = copy.deepcopy(y_outputs)
            random.shuffle(y_outputs_copy)
            fake_losses = [criterion(outputs, labels.float()).item() for outputs, labels in zip(y_outputs_copy, y_true)]
            e_losses.append(sum(fake_losses) / len(fake_losses))
        # e_loss=sum(e_losses)/len(e_losses)
        # delta_loss=[true_loss-e_loss for true_loss in true_losses]
        delta_losses = [true_loss - e_loss for true_loss, e_loss in zip(true_losses, e_losses)]
        delta_loss = sum(delta_losses)
        if delta_loss < 0:
            selected_genes.append(gene_name)

    with open('selected_gene.txt', 'w') as w:
        w.write(','.join(selected_genes))


if __name__ == '__main__':
    main()
