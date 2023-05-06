# -*- encoding: utf-8 -*-
"""
@File Name      :   merge_participants.py
@Create Time    :   2022/11/25 17:39
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


@click.command()
@click.argument('dir_path')
@click.option('-o', '--output_file_path', default='filtered_participant.csv', help='合并后的文件名')
@click.option('-i', '--limit_file_path', help='限制的参与者的文件路径')
def main(dir_path: str, output_file_path: str, limit_file_path: str):
    df = pd.DataFrame(columns=['Participant ID'])
    participant_file_names = [participant_file_name for participant_file_name in os.listdir(dir_path) if
                              participant_file_name.endswith('.csv') or participant_file_name.endswith('.txt')]
    if limit_file_path:
        participants_df = pd.read_csv(limit_file_path)
        participant_ids = participants_df['ID'].tolist()
        participant_file_names = [participant_file_name for participant_file_name in participant_file_names
                                  if participant_file_name.split('_')[0] in participant_ids]
    count = 0
    for participant_file_name in participant_file_names:
        participant_id = participant_file_name.split('_')[0]
        participant_file_path = os.path.join(dir_path, participant_file_name)
        # 如果是单人文件，一个文件的大小是可以处理的过来的，不用分块读取
        participant_df = pd.read_csv(participant_file_path, sep='\t')
        df.loc[count, 'Participant ID'] = participant_id
        for index, row in participant_df.iterrows():
            df.loc[count, row['SNP ID']] = row['Count']
        count += 1
    df = df.fillna(0)
    df.to_csv(output_file_path, index=False)
    return df


if __name__ == '__main__':
    main()
