# -*- encoding: utf-8 -*-
"""
@File Name      :   gene.py   
@Create Time    :   2023/9/11 18:28
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

import torch

from .compute.normal import min_max_normalization


def handle_gene_file(gene_file_path, gene_freq: list[dict] = None, normalization: bool = False) -> torch.Tensor:
    gene_file = open(gene_file_path, 'r')
    gene_data = gene_file.read().strip().split(',')[1:]
    gene_file.close()
    if gene_freq:
        freq_gene_data = [gene_freq[index][gene] for index, gene in enumerate(gene_data)]
        gene_data = torch.tensor([gene for gene in freq_gene_data], dtype=torch.float)
    else:
        gene_data = torch.tensor([int(gene) for gene in gene_data], dtype=torch.float)
        # gene_data = torch.tensor([snp - 1 if (snp := int(gene)) == 0 else snp for gene in gene_data], dtype=torch.float32)
        # gene_data = torch.tensor([int(gene)+1 for gene in gene_data], dtype=torch.float)
    if normalization:
        gene_data = min_max_normalization(gene_data)
    return gene_data


def handle_gene_file_bert(gene_file_path):
    gene_file = open(gene_file_path, 'r')
    gene_data = gene_file.read().strip().split(',')[1:]
    gene_file.close()
    return ' '.join(gene_data)
