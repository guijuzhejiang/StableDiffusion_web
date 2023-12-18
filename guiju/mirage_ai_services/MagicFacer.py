# coding=utf-8
# @Time : 2023/12/15 下午3:15
# @File : magic_cert.py
import glob
import os
import urllib.parse
from datetime import datetime

import cv2
from PIL import Image

from utils.global_vars import CONFIG


class MagicFacer(object):
    operator = None
    sd_model_name = None

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        pic_name = kwargs['pic_name']

        _input_src_image = cv2.imread(kwargs['input_image_paths'][0])
        _input_tgt_image = cv2.imread(kwargs['input_image_paths'][1])

        if self.operator.update_progress(60):
            return {'success': True}
        src_faces = self.operator.face_analysis.get(_input_src_image)
        if len(src_faces) != 1:
            # return {'success': False, 'result': '未检测到人脸'}
            return {'success': False, 'result': 'backend.magic-facer.error.no-face0'}
        tgt_faces = self.operator.face_analysis.get(_input_tgt_image)
        if len(tgt_faces) != 1:
            # return {'success': False, 'result': '未检测到人脸'}
            return {'success': False, 'result': 'backend.magic-facer.error.no-face1'}

        res = self.operator.swapper.get(_input_tgt_image, tgt_faces[0], src_faces[0], paste_back=True)
        # storage img
        dir_path = os.path.join(CONFIG['storage_dirpath']['user_facer_dir'], user_id)
        os.makedirs(dir_path, exist_ok=True)

        img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
        cv2.imwrite(os.path.join(dir_path, img_fn), res)

        if self.operator.update_progress(80):
            return {'success': True}
        # face fix
        gfpgan_weight = 0.5
        scales = 1
        codeformer_weight = 0
        codeformer_visibility = 0
        args = (0, scales, None, None, True, 'ESRGAN_4x', 'None',
                0, gfpgan_weight,
                codeformer_visibility, codeformer_weight)
        self.operator.devices.torch_gc()
        pp = self.operator.scripts_postprocessing.PostprocessedImage(Image.open(os.path.join(dir_path, img_fn)))
        self.operator.scripts.scripts_postproc.run(pp, args)
        # pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
        self.operator.devices.torch_gc()

        return [pp.image]
