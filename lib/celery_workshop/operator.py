import json
import sys


class Operator(object):
    def __init__(self):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def operation(self, *args, **kwargs):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

