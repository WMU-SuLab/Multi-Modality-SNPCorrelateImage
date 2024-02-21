# -*- encoding: utf-8 -*-
"""
@File Name      :   count_gene_num.py   
@Create Time    :   2023/7/24 17:01
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

import click


@click.command()
@click.argument('input_gene_file_path', type=click.Path(exists=True))
def main(input_gene_file_path: str):
    with open(input_gene_file_path, 'r') as f:
        text = f.read()
        print(len(text.split(',')[1:]))


if __name__ == '__main__':
    """
    示例：python data_pretreatment/gene/count_gene_num.py \
    work_dirs/data/gene/students_snps_all_frequency_0.001/label_20230919183434/selected_genes_percentile_30/columns.csv
    """
    main()
