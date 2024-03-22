# coding=utf-8
# @Time : 2023/8/10 下午1:48
# @File : atest.py
# -*- encoding: utf-8 -*-
'''
@File    :   ocr_process.py
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/21 上午10:21   ray      1.0         None
'''

# import lib
import datetime
import importlib
import os
import traceback

import ujson

from lib.common.common_util import logging
from lib.celery_workshop.operator import Operator


class OperatorSora(Operator):
    num = 1
    cache = True
    cuda = True
    enable = False
    celery_task_name = 'sora_task'

    def __init__(self, gpu_idx=0):
        Operator.__init__(self)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        print("use gpu:" + str(gpu_idx))

        print('start init OperatorSora')
        super().__init__()

        self.Image2Video = getattr(importlib.import_module('scripts.gradio.i2v_test_zzg'), 'Image2Video')(resolution='576_1024')

    def __call__(self, *args, **kwargs):
        try:
            super().__call__(*args, **kwargs)

            # log start
            print(f"{str(datetime.datetime.now())} sora operation start {kwargs['input_image']} !!!!!!!!!!!!!!!!!!!!!!!!!!")
            user_id = kwargs['user_id'][0]
            params = ujson.loads(kwargs['params'][0])
            origin = kwargs['origin']

            res = self.Image2Video.get_image(kwargs['input_image'], 'man fishing in a boat at sunset')

            return {'success': True, 'result': res}

        except Exception:
            logging(
                f"[ocr predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        return {'success': False, 'result': 'backend.generate.error.failed'}
