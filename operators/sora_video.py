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
import secrets
import traceback
import urllib.parse

import numpy as np
import ujson
from PIL import Image

from lib.common.common_util import logging
from lib.celery_workshop.operator import Operator
from utils.global_vars import CONFIG


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
            proceed_mode = kwargs['mode'][0]
            origin = kwargs['origin']

            dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], proceed_mode, user_id)
            os.makedirs(dir_path, exist_ok=True)
            video_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.mp4"

            res = self.Image2Video.get_image(np.array(Image.open(kwargs['input_image'])), params['prompt'], os.path.join(dir_path, video_fn), fs=int(params['video_len']), seed=secrets.randbelow(10000) + 1)
            url_fp = f"{'localhost/service' + str(CONFIG['server']['port']) if CONFIG['local'] else f'{origin}/service'}/user/video/fetch?path={video_fn}&uid={urllib.parse.quote(user_id)}&category={proceed_mode}"

            return {'success': True, 'result': url_fp}

        except Exception:
            logging(
                f"[ocr predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        return {'success': False, 'result': 'backend.generate.error.failed'}
