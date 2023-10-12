# -*- encoding: utf-8 -*-
"""
@File Name      :   vcf_filter_snps_with_gene_regions.py
@Create Time    :   2023/8/30 22:03
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

import gzip
import json
import os
from collections import defaultdict
from operator import itemgetter
from time import time

import click

from base import snp_info_columns, regularize_rules


@click.command()
@click.argument('snps_genes_file_path', type=click.Path(exists=True))
@click.argument('vcf_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
def main(snps_genes_file_path, vcf_dir_path, output_dir_path):
    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    snps_genes_file_path = os.path.abspath(snps_genes_file_path)
    with open(snps_genes_file_path, 'r') as f:
        snps_genes = json.load(f)
    vcf_dir_path = os.path.abspath(vcf_dir_path)
    gene_file_names = [file_name for file_name in os.listdir(vcf_dir_path) if file_name.endswith('.vcf.gz')]
    gene_file_paths = [os.path.join(vcf_dir_path, file_name) for file_name in gene_file_names]

    sep = "\t"
    skip_rows = 0
    with gzip.open(gene_file_paths[0], 'rt') as f:
        stop = False
        while not stop:
            line = f.readline()
            if line.startswith('##'):
                skip_rows += 1
            else:
                stop = True
                column_line = line
        columns = column_line.strip().split(sep)
        participant_columns = list(set(columns) - set(snp_info_columns))
        filtered_columns = ['#CHROM', 'POS', *participant_columns]
        filtered_columns_index = [columns.index(column) for column in filtered_columns]
        participant_files = []
        for participant_id in participant_columns:
            file = open(os.path.join(output_dir_path, f'{participant_id}.csv'), 'w')
            file.write(f'gene,snp_id,val\n')
            participant_files.append(file)

    for input_file_path in gene_file_paths:
        print(f'正在处理文件{input_file_path}')
        count = 0
        start = time()
        with gzip.open(input_file_path, 'rt') as f:
            for _ in range(skip_rows + 1):
                f.readline()
            for line in f:
                columns = line.strip().split(sep)
                filtered_columns = list((itemgetter(*filtered_columns_index)(columns)))
                chromosome = filtered_columns[0]
                pos = filtered_columns[1]
                snp_id = f'{chromosome}:{pos}'
                if snps_genes.get(snp_id, None):
                    count += 1
                else:
                    continue
                gene_name = snps_genes[snp_id]
                regularized_columns = [regularize_rules[participant_column.split(':')[0]] for participant_column in
                                       filtered_columns[2:]]
                for participant_file, regularized_column in zip(participant_files, regularized_columns):
                    participant_file.write(f'{gene_name},{snp_id},{regularized_column}\n')
                if count % 1000 == 0:
                    for file in participant_files:
                        file.flush()
                    print(f'count:{count},time:{time() - start}')
    for file in participant_files:
        file.close()


if __name__ == '__main__':
    """
    从vcf文件中制作gwis需要的participant_id.csv
    """
    main()
