# -*- encoding: utf-8 -*-
"""
@File Name      :   time.py
@Create Time    :   2023/3/1 16:47
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

from datetime import datetime


def datetime_now() -> datetime:
    # 获取当前时间
    return datetime.now()


def datetime_now_str() -> str:
    # 获取处理后的时间
    return datetime_now().strftime('%Y%m%d%H%M%S')


def datetime_now_str_multi_train() -> str:
    # 为了防止多GPU训练的时候出错，所以去掉秒
    return datetime_now().strftime('%Y%m%d%H%M')
