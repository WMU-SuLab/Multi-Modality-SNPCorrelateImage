# -*- encoding: utf-8 -*-
"""
@File Name      :   lists.py
@Create Time    :   2023/3/1 16:08
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

from operator import itemgetter


def extract_list_from_index(source_list: list, index_list: list) -> list:
    """
    从列表中提取指定索引的元素
    :param source_list:
    :param index_list:
    :return:
    """
    return list((itemgetter(*index_list)(source_list)))