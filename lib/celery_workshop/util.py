# coding=utf-8
# @Time : 2023/8/10 上午11:25
# @File : util.py
import inspect
import os
from importlib import import_module

from lib.celery_workshop.operator import Operator
from lib.celery_workshop.wokrshop import WorkShop


def load_workshops():
    operators = []
    # modules = import_module(f'operators')
    for fn in os.listdir('operators'):
        if fn[-3:] == '.py' and fn != "__init__.py":
            module = import_module(f'operators.{fn[:-3]}')
            all_classes = inspect.getmembers(inspect.getmodule(module), inspect.isclass)
            for c_name, c_obj in all_classes:
                if c_obj is not Operator and issubclass(c_obj, Operator):
                    workshop = WorkShop(c_obj)
                    operators.append(workshop)

        # for op_name in dir(modules):
        #     op_obj = getattr(modules, op_name)
        #     if isinstance(op_obj, type):
        #         workshop = WorkShop(op_obj)
        #         operators.append(workshop)
    else:
        return operators
