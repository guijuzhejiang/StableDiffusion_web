# coding=utf-8
# @Time : 2023/12/15 下午3:28
# @File : magic_ai_conductor.py
import importlib
import os


class MagicAiConductor(object):
    operator = None
    task_dict = {}
    modules_dir = "guiju/mirage_ai_services"

    def __init__(self, operator):
        # load all
        for filename in os.listdir(self.modules_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]  # Remove the ".py" extension
                module = importlib.import_module(f"{self.modules_dir.replace('/', '.')}.{module_name}")

                obj = getattr(module, module_name)
                if isinstance(obj, type):
                    self.task_dict[module_name.replace('Magic', '').lower()] = obj(operator)

        else:
            self.operator = operator

    def __call__(self, *args, **kwargs):
        proceed_mode = args[0]

        return self.task_dict[proceed_mode](**kwargs)

