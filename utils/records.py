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

import torch
from sklearn.metrics import roc_curve
from torch.utils.tensorboard import SummaryWriter

from .metrics import count_metrics_binary_classification


def experiments_record(
        record_file_path, train_dir_prefix, model_name, dataset_dir_path, snp_numbers,
        epochs, batch_size, lr, step_size, criterion, gamma, save_interval, log_interval, remarks=None):
    with open(record_file_path, 'a') as f:
        f.write(f"train_dir_prefix: {train_dir_prefix}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"dataset_dir_path: {dataset_dir_path}\n")
        f.write(f"snp_numbers: {snp_numbers}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"step_size: {step_size}\n")
        f.write(f"loss: {criterion.__class__.__name__}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"save_interval: {save_interval}\n")
        f.write(f"log_interval: {log_interval}\n")
        f.write(f"remarks: {remarks}\n")


def train_epoch_record(
        epoch_loss, all_metrics, net, optimizer, epoch, epochs, phase,
        writer, log_interval, best_f1, best_model_wts, best_model_checkpoint_path, since):
    acc, mcc, precision, recall, f1, fpr, tpr, ks, sp, auc_score = all_metrics
    # 得到最好的模型，需要自己定义哪种情况下指标最好
    if phase == 'valid' and f1 > best_f1:
        best_f1 = f1
        best_model_wts = copy.deepcopy(net.state_dict())
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
    time_elapsed = time.time() - since
    print(
        f'Epoch {epoch + 1}/{epochs} | {phase} | Loss: {epoch_loss:.4f} | best F1: {best_f1}\n'
        f'ACC: {acc:.4f} | MCC: {mcc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n'
        f'FPR: {fpr:.4f} | TPR: {tpr:.4f} | KS: {ks:.4f} | SP: {sp:.4f} | AUC: {auc_score:.4f}\n'
        f'Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    )
    return best_f1, best_model_wts


def test_metrics_record(y_true, y_pred, y_score, writer: SummaryWriter):
    # fig=plt.figure()
    # P-R曲线
    # precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # plt.title('P-R')
    # plt.xlabel('precision')
    # plt.ylabel('recall')
    # plt.plot(precision, recall)
    # writer.add_figure('P-R', plt.gcf())
    # writer.add_pr_curve('P-R', y_true, y_pred)
    # ROC曲线，使用 score 预测概率而不是预测类别，使得 ROC 曲线更平滑
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # plt.title('ROC')
    # plt.xlabel('FPR')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.ylabel('TPR')
    # plt.plot(fpr, tpr, 'b')
    # writer.add_figure('ROC', plt.gcf())
    # writer.add_figure('ROC', fig)
    for x, y in zip(fpr, tpr):
        writer.add_scalar('ROC', y, x)
    acc, mcc, precision, recall, f1, fpr, tpr, ks, sp, auc_score = count_metrics_binary_classification(
        y_true, y_pred, y_score)
    writer.add_text('Test Accuracy', str(acc))
    writer.add_text('Test MCC', str(mcc))
    writer.add_text('Test Precision', str(precision))
    writer.add_text('Test Recall', str(recall))
    writer.add_text('Test F1', str(f1))
    writer.add_text('Test FPR', str(fpr))
    writer.add_text('Test TPR', str(tpr))
    writer.add_text('Test KS', str(ks))
    writer.add_text('Test SP', str(sp))
    writer.add_text('Test AUC', str(auc_score))
    writer.flush()
    print(f'Acc: {acc:.4f} | MCC:{mcc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n')
    print(f'FPR: {fpr:.4f} | TPR: {tpr:.4f} | KS: {ks:.4f} | SP: {sp:.4f} | AUC: {auc_score:.4f}\n')
