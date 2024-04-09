# -*- encoding: utf-8 -*-
"""
@File Name      :   base.py
@Create Time    :   2022/11/29 11:24
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

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
disease_data_dir = os.path.join(data_dir, 'disease')
# 定义文件名称
all_participants_file_name = 'participants_label.csv'
all_participants_file_path = os.path.join(disease_data_dir, all_participants_file_name)
vcf_rename_rules_file_name = 'vcf_rename_rules.txt'
vcf_rename_rules_file_path = os.path.join(disease_data_dir, vcf_rename_rules_file_name)
AMD_file_name = 'AMD.csv'
AMD_file_path = os.path.join(disease_data_dir, AMD_file_name)
no_AMD_file_name = f'no_{AMD_file_name}'
no_AMD_file_path = os.path.join(disease_data_dir, no_AMD_file_name)
DR_file_name = 'DR.csv'
DR_file_path = os.path.join(disease_data_dir, DR_file_name)
no_DR_file_name = f'no_{DR_file_name}'
no_DR_file_path = os.path.join(disease_data_dir, no_DR_file_name)
GLC_file_name = 'GLC.csv'
GLC_file_path = os.path.join(disease_data_dir, GLC_file_name)
no_GLC_file_name = f'no_{GLC_file_name}'
no_GLC_file_path = os.path.join(disease_data_dir, no_GLC_file_name)
RD_file_name = 'RD.csv'
RD_file_path = os.path.join(disease_data_dir, RD_file_name)
no_RD_file_name = f'no_{RD_file_name}'
no_RD_file_path = os.path.join(disease_data_dir, no_RD_file_name)
no_disease_file_name = f'no_disease.csv'
no_disease_file_path = os.path.join(disease_data_dir, no_disease_file_name)
