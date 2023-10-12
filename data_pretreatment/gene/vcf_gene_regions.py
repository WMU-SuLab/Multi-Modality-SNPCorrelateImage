# -*- encoding: utf-8 -*-
"""
@File Name      :   vcf_gene_regions.py
@Create Time    :   2023/8/30 14:59
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
import os.path
import re
from collections import defaultdict
from operator import itemgetter
from time import time

import click
import pandas as pd


@click.command()
@click.argument('gene_region_file_path', type=click.Path(exists=True))
@click.argument('vcf_dir_path', type=click.Path(exists=True))
def main(gene_region_file_path, vcf_dir_path):
    gene_region_file_path = os.path.abspath(gene_region_file_path)
    gene_region_file_dir_path = os.path.dirname(gene_region_file_path)
    gene_region_file_name = os.path.basename(gene_region_file_path)
    df = pd.read_csv(gene_region_file_path, sep='\t', header=None)
    df = df[[0, 2, 3, 4, 8]]
    gene_regions = defaultdict(dict)
    gene_regions_chromosome = defaultdict(list)
    for index, row in df.iterrows():
        chromosome = row[0]
        region_type = row[2]
        start = row[3]
        end = row[4]
        infos = row[8]
        if region_type == 'gene':
            search = re.search('gene_name "([A-Za-z0-9]+)"', infos)
            if not search:
                continue
            gene_name = search.group(1)
            gene_regions[chromosome][f'{start}_{end}'] = gene_name
            gene_regions_chromosome[chromosome].append((start, end))
    with open(os.path.join(gene_region_file_dir_path, f'{gene_region_file_name}.gene_regions.json'), 'w') as f:
        json.dump(gene_regions, f)
    for item in gene_regions_chromosome.values():
        item.sort()
    gene_regions_chromosome_df = {gene: pd.DataFrame(region, columns=['start', 'end']) for gene, region in
                                  gene_regions_chromosome.items()}

    gene_dir_path = os.path.abspath(vcf_dir_path)
    gene_file_names = [file_name for file_name in os.listdir(gene_dir_path) if file_name.endswith('.vcf.gz')]
    gene_file_paths = [os.path.join(gene_dir_path, file_name) for file_name in gene_file_names]

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
        filtered_columns = ['#CHROM', 'POS']
        filtered_columns_index = [columns.index(filtered_column) for filtered_column in filtered_columns]
    snps_gene_regions = {}
    gene_regions_snps = defaultdict(list)
    for gene_file_path in gene_file_paths:
        print(f'正在处理文件{gene_file_path}')
        count = 0
        start = time()
        with gzip.open(gene_file_path, 'rt') as f:
            for _ in range(skip_rows + 1):
                f.readline()
            for line in f:
                count += 1
                columns = line.strip().split(sep)
                filtered_columns = list((itemgetter(*filtered_columns_index)(columns)))
                chromosome, pos = filtered_columns
                pos = int(pos)
                region_df = gene_regions_chromosome_df[chromosome]
                region_indexes = region_df[(region_df['start'] <= pos) & (region_df['end'] >= pos)].index.tolist()
                if region_indexes:
                    region_index = region_indexes[0]
                else:
                    continue
                region_start, region_end = gene_regions_chromosome[chromosome][region_index]
                gene_name = gene_regions[chromosome][f'{region_start}_{region_end}']
                snp_id = f'{chromosome}:{pos}'
                snps_gene_regions[snp_id] = gene_name
                gene_regions_snps[gene_name].append(snp_id)
                if count % 1000 == 0:
                    print(f'row count:{count}, time:{time() - start}')
    with open(os.path.join(gene_dir_path, f'{gene_region_file_name}.snps_genes.json'), 'w') as f:
        json.dump(snps_gene_regions, f)
    with open(os.path.join(gene_dir_path, f'{gene_region_file_name}.genes_snps.json'), 'w') as f:
        json.dump(gene_regions_snps, f)

    with open(os.path.join(gene_dir_path, f'{gene_region_file_name}.gene_regions_info.json'), 'w') as f:
        json.dump({
            'gene_names': list(gene_regions_snps.keys()),
            'gene_region_snps_len': {gene_name: len(snps) for gene_name, snps in gene_regions_snps.items()}
        }, f)


if __name__ == '__main__':
    """
    从vcf中统计基因区域的信息，是vcf_filter_snps_with_gene_regions.py的前置脚本
    """
    main()
