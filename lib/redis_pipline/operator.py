import json
import sys


class Operator(object):
    first_op = False
    last_op = False

    def __init__(self):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def operation(self, *args, **kwargs):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")
        msg, = args
        return msg, json.loads(msg['json_msg'])

    def return_msg(self, package, data=None):
        if data is not None:
            package['json_msg'] = json.dumps(data)
        return package
