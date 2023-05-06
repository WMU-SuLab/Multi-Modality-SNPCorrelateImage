# -*- encoding: utf-8 -*-
"""
@File Name      :   metrics.py
@Create Time    :   2023/3/4 18:50
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

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def count_metrics_binary_classification(y_true, y_pred, y_score):
    """
    计算指标（二分类）
    :param y_true:
    :param y_pred:
    :param y_score:
    :return:
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    ks = abs(tpr - fpr)
    # sn = tp / (tp + fn)
    sp = 1 - fpr
    # sp=tn / (tn+fp)
    acc = accuracy_score(y_true, y_pred)
    # acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    precision = precision_score(y_true, y_pred)
    # p=tp / (tp + fp)
    recall = recall_score(y_true, y_pred)
    # r=tp / (tp + fn)
    f1 = f1_score(y_true, y_pred)
    # f1 = 2 * precision * recall / (precision + recall)
    # KS
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # ks = abs(fpr - tpr).max()
    # auc_score = auc(fpr, tpr)
    # 不绘制曲线可以直接用 roc_auc_score 函数计算AUC
    auc_score = roc_auc_score(y_true, y_score)
    return acc, mcc, precision, recall, f1, fpr, tpr, ks, sp, auc_score


def count_metrics_multi_classification(y_true, y_pred, y_score):
    """
    计算指标（多分类）
    :param y_true:
    :param y_pred:
    :param y_score:
    :return:
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # auc_score = auc(fpr, tpr)
    auc_score = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
    return acc, precision, recall, f1, auc_score
