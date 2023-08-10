# coding=utf-8
# @Time : 2023/8/10 上午11:25
# @File : util.py
from importlib import import_module

from lib.celery_workshop.wokrshop import WorkShop


def load_workshops():
    operators = []
    modules = import_module(f'operators')
    for op_name in dir(modules):
        op_obj = getattr(modules, op_name)
        if isinstance(op_obj, type):
            workshop = WorkShop(op_obj)
            operators.append(workshop)
    else:
        return operators
