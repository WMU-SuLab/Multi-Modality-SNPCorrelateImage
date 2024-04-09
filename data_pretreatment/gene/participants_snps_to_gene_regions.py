# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_frequency_snps_to_gene_regions.py   
@Create Time    :   2023/9/19 18:19
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
import os
from collections import defaultdict

import click
import tqdm


@click.command()
@click.argument('snps_gene_regions_file_path', type=click.Path(exists=True))
@click.argument('input_snps_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
def main(snps_gene_regions_file_path, input_snps_dir_path, output_dir_path):
    if output_dir_path and (not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path)):
        os.mkdir(output_dir_path)
    with open(snps_gene_regions_file_path, 'r') as f:
        snps_gene_regions = json.load(f)

    with open(os.path.join(input_snps_dir_path, f'columns.csv'), 'r') as f:
        snp_ids = f.read().split(',')[1:]
    gene_names = []
    index_gene_name = {}
    gene_regions_len = defaultdict(int)
    for index, snp_id in enumerate(snp_ids):
        gene_name = snps_gene_regions.get(f'chr{snp_id}', None)
        index_gene_name[index] = gene_name
        if gene_name:
            gene_names.append(gene_name)
            gene_regions_len[gene_name] += 1
    gene_names = list(set(list(filter(None, gene_names))))
    print(len(gene_names))
    gene_files = {}
    for gene_name in gene_names:
        gene_file = open(os.path.join(output_dir_path, f'{gene_name}.csv'), 'w')
        gene_file.write('participant_id,snp_id,val\n')
        gene_files[gene_name] = gene_file

    file_names = [file_name for file_name in os.listdir(input_snps_dir_path)
                  if file_name.endswith('.csv') and file_name != 'columns.csv']
    for file_name in tqdm.tqdm(file_names):
        participant_id = file_name.split('.')[0]
        file_path = os.path.join(input_snps_dir_path, file_name)
        with open(file_path, 'r') as f:
            snps = f.read().split(',')[1:]
            for index, snp in enumerate(snps):
                if gene_name := index_gene_name[index]:
                    gene_files[gene_name].write(f'{participant_id},{snp_ids[index]},{snp}\n')

    for gene_file in gene_files.values():
        gene_file.close()

    with open(os.path.join(output_dir_path, 'gene_regions_info.json'), 'w') as f:
        json.dump({
            'gene_names': gene_names,
            'gene_region_snps_len': gene_regions_len,
        }, f)


if __name__ == '__main__':
    """
    从已经过滤的snps数据中直接制作gene_regions结构的文件，而不经过
    vcf_gene_regions.py，
    vcf_filter_snps_with_gene_regions.py，
    participants_gene_regions_filter_with_chosen_snps.py，
    participants_gene_regions_snps_divide.py
    这四步流程
    
    示例：python data_pretreatment/gene/participants_frequency_snps_to_gene_regions.py \
    work_dirs/data/gene/filtered_alleles_vcf/all_snps_gene_regions.json \
    work_dirs/data/gene/students_snps_all_frequency_0.001/ \
    work_dirs/data/gene/students_snps_all_frequency_0.001/gene_regions
    """
    main()
