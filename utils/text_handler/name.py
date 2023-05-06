# -*- encoding: utf-8 -*-
"""
@File Name      :   name.py
@Create Time    :   2023/4/14 16:59
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

import re


def pascal_case_to_snake_case(pascal_case: str) -> str:
    """大驼峰（帕斯卡）转蛇形"""
    snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", pascal_case)
    return snake_case.lower().strip('_')
