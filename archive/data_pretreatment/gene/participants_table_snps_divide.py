# -*- encoding: utf-8 -*-
"""
@File Name      :   participants_gene_regions_snps_divide.py
@Create Time    :   2023/9/1 14:40
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

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, wait

import click
import pandas as pd
import tqdm


def mk_dir(dir_path: str) -> bool:
    """
    没有就创建这个文件夹，有就直接返回True
    """
    # 为了防止是WindowsPath而报错，先转换一下
    path = str(dir_path).strip()
    if not os.path.exists(path) or not os.path.isdir(path):
        try:
            os.makedirs(path)
            # os.mkdir(path)
        except Exception as e:
            print(str(e))
            return False
    return True


def list_to_n_group(list_to_group: list, n: int = 3) -> list:
    length = len(list_to_group)
    remainder = length % n
    if remainder == 0:
        step = length // n
    else:
        step = length // n + 1
    return [list_to_group[i:i + step] for i in range(0, len(list_to_group), step)]


def write_csv(file_path, df):
    df.to_csv(file_path, mode='a', index=False, header=False)


def handle_gene_files(gene_file_names, gene_dir_path, output_dir_path, gene_num):
    for file_name in gene_file_names:
        file_path = os.path.join(gene_dir_path, file_name)
        dir_path = os.path.join(output_dir_path, file_name.split('.')[0])
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        if gene_num and len(os.listdir(dir_path)) == gene_num:
            continue
        df = pd.read_csv(file_path)
        for gene, group_df in df.groupby('gene'):
            group_df[['snp_id', 'val']].to_csv(os.path.join(dir_path, f'{gene}.csv'), index=None)


@click.command()
@click.argument('input_dir_path', type=click.Path(exists=True))
@click.option('--output_dir_path', type=str, default='')
@click.option('--gene_num', type=int, default=0)
@click.option('--process_num', type=int, default=32)
@click.option('--method', type=click.Choice(['participants', 'gene_regions']), default='participants')
def main(input_dir_path, output_dir_path, gene_num, process_num, method):
    input_dir_path = os.path.abspath(input_dir_path)
    participant_file_names = [file_name for file_name in os.listdir(input_dir_path) if file_name.endswith('.csv')]

    if not output_dir_path:
        output_dir_path = input_dir_path
    if method == 'participants':
        output_dir_path = os.path.join(output_dir_path, 'participants')
        mk_dir(output_dir_path)
        participant_file_names_groups = list_to_n_group(participant_file_names, process_num)
        if process_num > multiprocessing.cpu_count():
            pool = multiprocessing.Pool(processes=process_num)
        else:
            pool = multiprocessing.Pool()
        for gene_file_names in participant_file_names_groups:
            pool.apply_async(handle_gene_files, (gene_file_names, input_dir_path, output_dir_path, gene_num))
        pool.close()
        pool.join()
    elif method == 'gene_regions':
        output_dir_path = os.path.join(output_dir_path, 'gene_regions')
        mk_dir(output_dir_path)
        df = pd.read_csv(os.path.join(input_dir_path, participant_file_names[0]))
        gene_regions = df['gene'].drop_duplicates().tolist()
        gene_regions_files = {}
        for gene_region in gene_regions:
            gene_region_file_path = os.path.join(output_dir_path, f'{gene_region}.csv')
            f = open(gene_region_file_path, 'w')
            f.write('participant_id,snp_id,val\n')
            f.close()
            f = open(gene_region_file_path, 'a')
            gene_regions_files[gene_region] = f
        count = 0
        for participant_file_name in tqdm.tqdm(participant_file_names):
            count += 1
            participant_id = participant_file_name.split('.')[0]
            participant_file_path = os.path.join(input_dir_path, participant_file_name)
            df = pd.read_csv(participant_file_path)
            # with ProcessPoolExecutor() as executor:
            with ThreadPoolExecutor() as executor:
                tasks = []
                for gene_region, group_df in df.groupby('gene'):
                    group_df = group_df[['snp_id', 'val']]
                    group_df['participant_id'] = participant_id
                    group_df = group_df[['participant_id', 'snp_id', 'val']]
                    task = executor.submit(write_csv, gene_regions_files[gene_region], group_df)
                    # task = executor.submit(write_csv, os.path.join(output_dir_path, f'{gene_region}.csv'), group_df)
                    tasks.append(task)
                wait(tasks)
            if count % 100 == 0:
                for gene_region, f in gene_regions_files.items():
                    f.flush()
        for gene_region, f in gene_regions_files.items():
            f.close()
    else:
        raise ValueError('invalid methods')


if __name__ == '__main__':
    """
    把participant_id.csv拆成以下两种形式选其一：
    单人文件夹，每个文件夹有很多基因命名的csv文件，好处是数据不冗余，坏处是训练速度极慢
    gene_regions文件，把所有人的数据放到以基因命名的文件中，数据虽然冗余，但是大文件读取快，过滤人也快，训练速度大幅度上升
    """
    main()
