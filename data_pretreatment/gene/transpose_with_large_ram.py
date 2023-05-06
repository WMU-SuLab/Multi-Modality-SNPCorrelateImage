# -*- encoding: utf-8 -*-
"""
@File Name      :   transpose_with_large_ram.py
@Create Time    :   2022/12/5 15:11
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

import os

import click
import pandas as pd

from base import snp_info_columns


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.option('-o', '--output_file_path', default='transposed_chromosomes.csv', help='合并后的文件名')
@click.option('-s', '--skip_rows', default=0, help='跳过的行数')
def main(input_file_path: str, output_file_path: str, skip_rows: int):
    input_file_path = os.path.abspath(input_file_path)
    output_file_path = os.path.abspath(output_file_path)
    output_dir_path = os.path.dirname(output_file_path)
    participants_dir_path = os.path.join(output_dir_path, 'participants')
    df = pd.read_csv(input_file_path, skiprows=skip_rows)
    participant_columns = list(set(df.columns) - set(snp_info_columns))
    snp_columns = df['ID'].tolist()
    # 对文件重新组织为以参与者为行的形式
    new_df = pd.DataFrame(columns=['Participant ID', *snp_columns])
    new_df['Participant ID'] = participant_columns
    new_df[snp_columns] = df[participant_columns].values.T
    new_df.to_csv(output_file_path, index=False)
    count = 1
    for index, row in new_df.iterrows():
        count += 1
        participant_df = row.to_frame().T
        participant_df.to_csv(os.path.join(participants_dir_path, f'{row["Participant ID"]}.csv'), index=False,
                              header=False)
        if count == 1:
            participant_df = participant_df.drop(participant_df.index)
            participant_df.to_csv(os.path.join(participants_dir_path, 'columns.csv'), index=False)


if __name__ == '__main__':
    """
    计算机内存足够大，一次性处理完所有的数据
    """
    main()
