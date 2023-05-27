# -*- encoding: utf-8 -*-
"""
@File Name      :   gene.py
@Create Time    :   2023/4/27 11:04
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

import torch


def handle_gene_file(gene_file_path, gene_freq: list[dict] = None):
    gene_file = open(gene_file_path, 'r')
    gene_data = gene_file.read().strip().split(',')[1:]
    gene_file.close()
    if gene_freq:
        freq_gene_data = [gene_freq[index][gene] for index, gene in enumerate(gene_data)]
        return torch.tensor([gene for gene in freq_gene_data], dtype=torch.float32)
    else:
        return torch.tensor([int(gene) for gene in gene_data], dtype=torch.float32)
