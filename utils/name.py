# -*- encoding: utf-8 -*-
"""
@File Name      :   name.py
@Create Time    :   2023/3/2 21:26
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


def generate_gene_file_name(participant_id: str) -> str:
    return f'{participant_id}.csv'


def get_label_participant_id(label_id: str) -> str:
    return label_id.split('-')[0]

def get_image_file_participant_id_instance(image_file_name: str) -> str:
    return os.path.splitext(image_file_name)[0].split('_')[0]

def get_image_file_participant_id(image_file_name: str) -> str:
    return get_image_file_participant_id_instance(image_file_name).split('-')[0]


def get_gene_file_participant_id(gene_file_name: str) -> str:
    return os.path.splitext(gene_file_name)[0].split('_')[0]
