# -*- encoding: utf-8 -*-
"""
@File Name      :   records.py
@Create Time    :   2023/4/10 10:13
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

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

from utils.compute.metrics import count_metrics_binary_classification


def experiments_record(
        record_file_path, train_dir_prefix, model_name, dataset_dir_path, snp_numbers,
        epochs, batch_size, lr, step_size, criterion, gamma, save_interval, log_interval, remarks=None):
    with open(record_file_path, 'a') as f:
        f.write(f"train_dir_prefix: {train_dir_prefix}\n"
                f"model_name: {model_name}\n"
                f"dataset_dir_path: {dataset_dir_path}\n"
                f"snp_numbers: {snp_numbers}\n"
                f"epochs: {epochs}\n"
                f"batch_size: {batch_size}\n"
                f"lr: {lr}\n"
                f"step_size: {step_size}\n"
                f"loss: {criterion.__class__.__name__}\n"
                f"gamma: {gamma}\n"
                f"save_interval: {save_interval}\n"
                f"log_interval: {log_interval}\n"
                f"remarks: {remarks}\n")


def train_epoch_record(
        epoch_loss, all_metrics, net, optimizer, epoch, epochs, phase,
        writer, log_interval, best_f1, best_model_wts, best_model_checkpoint_path, since):
    # 得到最好的模型，需要自己定义哪种情况下指标最好
    if phase == 'valid' and all_metrics['f1'] > best_f1:
        best_f1 = all_metrics['f1']
        best_model_wts = copy.deepcopy(net.state_dict())
        # 保存最好的模型
        torch.save({
            'epoch': epoch,
            'model': best_model_wts,
            'f1': best_f1,
            'optimizer': optimizer.state_dict()
        }, best_model_checkpoint_path)
    # 记录
    if epoch % log_interval == 0:
        writer.add_scalars('Loss', {
            phase: epoch_loss,
        }, epoch)
        writer.add_scalars('acc', {
            phase: all_metrics['acc'],
        }, epoch)
        # writer.add_scalars('ACC', {
        #     phase: all_metrics['ACC'],
        # }, epoch)
        writer.add_scalars('mcc', {
            phase: all_metrics['mcc'],
        }, epoch)
        # writer.add_scalars('MCC', {
        #     phase: all_metrics['MCC'],
        # }, epoch)
        writer.add_scalars('precision', {
            phase: all_metrics['precision'],
        }, epoch)
        # writer.add_scalars('Precision', {
        #     phase: all_metrics['Precision'],
        # }, epoch)
        writer.add_scalars('recall', {
            phase: all_metrics['recall'],
        })
        # writer.add_scalars('Recall', {
        #     phase: all_metrics['Recall'],
        # }, epoch)
        writer.add_scalars('f1', {
            phase: all_metrics['f1'],
        })
        # writer.add_scalars('F1', {
        #     phase: all_metrics['F1'],
        # }, epoch)
        writer.add_scalars('tpr', {
            phase: all_metrics['tpr'],
        })
        writer.add_scalars('fpr', {
            phase: all_metrics['fpr'],
        })
        writer.add_scalars('ks', {
            phase: all_metrics['ks'],
        }, epoch)
        # writer.add_scalars('KS', {
        #     phase: all_metrics['KS'],
        # }, epoch)
        writer.add_scalars('sp', {
            phase: all_metrics['sp'],
        }, epoch)
        # writer.add_scalars('SP', {
        #     phase: all_metrics['SP'],
        # }, epoch)
        writer.add_scalars('auc', {
            phase: all_metrics['auc_score'],
        }, epoch)
        # writer.add_scalars('AUC', {
        #     phase: all_metrics['AUC'],
        # }, epoch)
        writer.flush()
    time_elapsed = time.time() - since
    # print(
    #     f"Epoch {epoch + 1}/{epochs} | {phase} | Loss: {epoch_loss:.4f} | best F1: {best_f1}\n"
    #     f"acc: {all_metrics['acc']:.4f} | ACC: {all_metrics['ACC']:.4f} | mcc: {all_metrics['mcc']:.4f} | \n"
    #     f"MCC: {all_metrics['MCC']:.4f} | precision: {all_metrics['precision']:.4f} | Precision: {all_metrics['Precision']:.4f} | \n"
    #     f"recall: {all_metrics['recall']:.4f} | Recall: {all_metrics['Recall']:.4f} | f1: {all_metrics['f1']:.4f} | \n"
    #     f"F1: {all_metrics['F1']:.4f} | tpr: {all_metrics['tpr']:.4f} | fpr: {all_metrics['fpr']:.4f} | \n"
    #     f"ks: {all_metrics['ks']:.4f} | KS: {all_metrics['KS']:.4f} | sp: {all_metrics['sp']:.4f} | \n"
    #     f"SP: {all_metrics['SP']:.4f} | auc: {all_metrics['auc_score']:.4f} | AUC: {all_metrics['AUC']:.4f}\n"
    #     f"Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    # )
    print(
        f"Epoch {epoch + 1}/{epochs} | {phase} | Loss: {epoch_loss:.4f} | best F1: {best_f1}\n"
        f"acc: {all_metrics['acc']:.4f} | precision: {all_metrics['precision']:.4f} | recall: {all_metrics['recall']:.4f} | \n"
        f"f1: {all_metrics['f1']:.4f} | mcc: {all_metrics['mcc']:.4f} | auc: {all_metrics['auc_score']:.4f}  | \n"
        f"tpr: {all_metrics['tpr']:.4f} | fpr: {all_metrics['fpr']:.4f} | ks: {all_metrics['ks']:.4f} | sp: {all_metrics['sp']:.4f}\n"
        f"Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    return all_metrics['f1'], best_f1, best_model_wts


def test_metrics_record(y_true, y_pred, y_score, writer: SummaryWriter):
    all_metrics = count_metrics_binary_classification(
        y_true, y_pred, y_score)
    plt.figure()
    # P-R曲线
    # tensorboard 原生方法
    writer.add_pr_curve('p-r', np.array(y_true), np.array(y_pred), global_step=0)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    plt.title('P-R')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.plot(precision, recall, marker='.')
    writer.add_figure('P-R', plt.gcf())
    writer.add_text('Test P-R-precision', str(precision))
    writer.add_text('Test P-R-recall', str(recall))
    # ROC曲线，使用 score 预测概率而不是预测类别，使得 ROC 曲线更平滑
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % all_metrics['auc_score'])
    writer.add_figure('ROC', plt.gcf())
    # 使用 tensorboard 直接记录
    for x, y in zip(fpr, tpr):
        writer.add_scalar('roc', y, x)
    writer.add_text('Test ROC-fpr', str(fpr))
    writer.add_text('Test ROC-tpr', str(tpr))
    # 记录 Test 数据集指标
    writer.add_text('Test acc', str(all_metrics['acc']))
    writer.add_text('Test ACC', str(all_metrics['ACC']))
    writer.add_text('Test mcc', str(all_metrics['mcc']))
    writer.add_text('Test MCC', str(all_metrics['MCC']))
    writer.add_text('Test precision', str(all_metrics['precision']))
    writer.add_text('Test Precision', str(all_metrics['Precision']))
    writer.add_text('Test recall', str(all_metrics['recall']))
    writer.add_text('Test Recall', str(all_metrics['Recall']))
    writer.add_text('Test f1', str(all_metrics['f1']))
    writer.add_text('Test F1', str(all_metrics['F1']))
    writer.add_text('Test tpr', str(all_metrics['tpr']))
    writer.add_text('Test fpr', str(all_metrics['fpr']))
    writer.add_text('Test ks', str(all_metrics['ks']))
    writer.add_text('Test KS', str(all_metrics['KS']))
    writer.add_text('Test sp', str(all_metrics['sp']))
    writer.add_text('Test SP', str(all_metrics['SP']))
    writer.add_text('Test auc', str(all_metrics['auc_score']))
    writer.add_text('Test AUC', str(all_metrics['AUC']))
    writer.flush()
    # print(f"Test Metrics: \n"
    #       f"acc: {all_metrics['acc']} | ACC: {all_metrics['ACC']:.4f} | \n"
    #       f"mcc: {all_metrics['mcc']} | MCC: {all_metrics['MCC']:.4f} | \n"
    #       f"precision: {all_metrics['precision']} | Precision: {all_metrics['Precision']:.4f} | \n"
    #       f"recall: {all_metrics['recall']} | Recall: {all_metrics['Recall']:.4f} | \n"
    #       f"f1: {all_metrics['f1']} | F1: {all_metrics['F1']:.4f} | \n"
    #       f"tpr: {all_metrics['tpr']} | fpr: {all_metrics['fpr']} | \n"
    #       f"ks: {all_metrics['ks']} | KS: {all_metrics['KS']:.4f} | \n"
    #       f"sp: {all_metrics['sp']} | SP: {all_metrics['SP']:.4f} | \n"
    #       f"auc: {all_metrics['auc_score']} | AUC: {all_metrics['AUC']:.4f}\n")
    print(f"Test Metrics: \n"
          f"ACC: {all_metrics['ACC']:.4f} | Precision: {all_metrics['Precision']:.4f} | Recall: {all_metrics['Recall']:.4f} | \n"
          f"F1: {all_metrics['F1']:.4f} | MCC: {all_metrics['MCC']:.4f} | AUC: {all_metrics['AUC']:.4f} | \n"
          f"tpr: {all_metrics['tpr']:.4f} | fpr: {all_metrics['fpr']:.4f} | KS: {all_metrics['KS']:.4f} | SP: {all_metrics['SP']:.4f}\n")
