# -*- encoding: utf-8 -*-
"""
@File Name      :   build_new_vcf_headers.py
@Create Time    :   2022/11/29 19:37
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

import pandas as pd

from base import all_participants_file_path, vcf_rename_rules_file_name


def build_replace_dict():
    """
    Build a dictionary to replace the headers of the data frame.
    :return:
    """
    df = pd.read_csv(all_participants_file_path)
    participant_ids = df['ID']
    with open(vcf_rename_rules_file_name, 'w') as f:
        lines = [f'{participant_id}_{participant_id} {participant_id}\n' for participant_id in participant_ids]
        f.writelines(lines)


if __name__ == '__main__':
    build_replace_dict()