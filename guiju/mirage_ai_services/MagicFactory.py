# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import datetime
import glob
import os
import random
import string

import numpy as np
from PIL import Image

from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion


class MagicFactory(object):
    operator = None
    # sd_model_name = 'dreamshaper_8'

    # denoising_strength_min = 0.5
    # denoising_strength_max = 1

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        _input_image = kwargs['input_image']
        pic_name = kwargs['pic_name']

        _character = str(params['character'])
        _place = int(params['place'])
        _clothing = int(params['clothing'])
        _batch_size = int(params['batch_size'])

        # save cache face img
        _input_image.save(f'tmp/{self.__class__.__name__}_origin_{pic_name}_save.png')
        # _input_image = _input_image.convert('RGBA')

        _input_image_width, _input_image_height = _input_image.size

        if self.operator.update_progress(10):
            return {'success': True}

        # parse face
        face_boxes = self.operator.facer.detect_face(_input_image)
        if len(face_boxes) == 0:
            # return {'success': False, 'result': '未检测到人脸'}
            return {'success': False, 'result': 'backend.magic-factory.error.no-face'}

        elif len(face_boxes) > 1:
            # return {'success': False, 'result': '检测到多个人脸，请上传一张单人照'}
            return {'success': False, 'result': 'backend.magic-factory.error.multi-face'}

        else:
            if self.operator.update_progress(30):
                return {'succ'
                        'ess': True}

            # prompt
            positive_prompt = ''
            negative_prompt = ''
            res = self.operator.faceid_predictor(np.array(_input_image), positive_prompt, negative_prompt, _batch_size)

        return res
