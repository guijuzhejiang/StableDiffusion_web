# coding=utf-8
# @Time : 2023/12/15 下午3:15
# @File : magic_cert.py
import glob
import os
import urllib.parse
from datetime import datetime

from PIL import Image

from utils.global_vars import CONFIG


class MagicCert(object):
    operator = None
    sd_model_name = None

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        _input_image = kwargs['input_image']
        pic_name = kwargs['pic_name']

        _bg_color = str(params['bg_color'])
        _output_aspect = float(params['aspect'])

        # save cache face img
        _input_image.save(f'tmp/{self.__class__.__name__}_origin_{pic_name}_save.png')
        _input_image = _input_image.convert('RGBA')

        _input_image_width, _input_image_height = _input_image.size

        if self.operator.update_progress(10):
            return {'success': True}
        # parse face
        face_boxes = self.operator.facer.detect_face(_input_image)
        if len(face_boxes) == 0:
            # return {'success': False, 'result': '未检测到人脸'}
            return {'success': False, 'result': 'backend.magic-avatar.error.no-face'}

        elif len(face_boxes) > 1:
            # return {'success': False, 'result': '检测到多个人脸，请上传一张单人照'}
            return {'success': False, 'result': 'backend.magic-avatar.error.multi-face'}

        else:
            if self.operator.update_progress(30):
                return {'success': True}

            # segment person
            sam_person_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, 'person', 0.3,
                                                                   _input_image)
            padding = 4
            resize_mask = sam_person_result[1].resize((_input_image_width - padding, _input_image_height - padding))
            padding_mask = Image.new("RGB", _input_image.size, (0, 0, 0, 1))
            padding_mask.paste(resize_mask, (int(padding / 2), int(padding / 2)))
            padding_mask = padding_mask.convert('1')

            padding_sam = Image.new('RGBA', _input_image.size)
            padding_sam.paste(sam_person_result[2], (0, 0), padding_mask)
            sam_person_result[2] = padding_sam

        if self.operator.update_progress(60):
            return {'success': True}

        # get max area face box
        face_box = face_boxes[0]
        face_width = face_box[2] - face_box[0]
        face_height = face_box[3] - face_box[1]

        padding_ratio = 1
        face_box[0] = face_box[0] - int(face_width * padding_ratio)
        if face_box[0] < 0:
            face_box[0] = 0
        face_box[1] = face_box[1] - int(face_height * padding_ratio)
        if face_box[1] < 0:
            face_box[1] = 0
        face_box[2] = face_box[2] + int(face_width * padding_ratio)
        if face_box[2] >= _input_image_width:
            face_box[2] = _input_image_width - 1
        face_box[3] = face_box[3] + int(face_height * padding_ratio)
        if face_box[3] >= _input_image_height:
            face_box[3] = _input_image_height - 1

        _cur_width = face_box[2] - face_box[0]
        _cur_height = face_box[3] - face_box[1]
        _cur_aspect = _cur_width / _cur_height

        # 计算应该添加的填充量
        if _cur_aspect > _output_aspect:
            # 需要添加垂直box
            target_height = int(_cur_width / _output_aspect)

            if int((target_height - _cur_height) / 2) + face_box[3] <= _input_image_height:
                face_box[3] = int((target_height - _cur_height) / 2) + face_box[3]
                _cur_height = face_box[3] - face_box[1]
            if face_box[1] - int((target_height - _cur_height) / 2) >= 0:
                face_box[1] = face_box[1] - int((target_height - _cur_height) / 2)
                _cur_height = face_box[3] - face_box[1]

            left = face_box[0]
            if target_height > face_box[3] - face_box[1]:
                top = int((target_height - _cur_height) / 2)

            else:
                top = 0

            canvas_wh = (_cur_width, target_height)
        else:
            # 需要添加水平box
            target_width = int(_cur_height * _output_aspect)

            if face_box[0] - int((target_width - _cur_width) / 2) >= 0:
                face_box[0] = face_box[0] - int((target_width - _cur_width) / 2)
                _cur_width = face_box[2] - face_box[0]
            if face_box[2] + int((target_width - _cur_width) / 2) <= _input_image_width:
                face_box[2] = face_box[2] + int((target_width - _cur_width) / 2)
                _cur_width = face_box[2] - face_box[0]

            top = face_box[1]
            if target_width > face_box[2] - face_box[0]:
                left = int((target_width - _cur_width) / 2)

            else:
                left = 0

            canvas_wh = (target_width, _cur_height)

        # new canvas
        cert_res = Image.new("RGBA", canvas_wh, _bg_color)
        foreground = sam_person_result[2].crop(face_box)
        cert_res.paste(foreground, (left, top), mask=foreground)

        # storage img
        dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], 'cert')

        os.makedirs(dir_path, exist_ok=True)
        img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
        # cert_res.convert("RGB").save(os.path.join(dir_path, img_fn), format="jpeg", quality=80,
        #                             lossless=True)
        # face fix
        gfpgan_weight = 0.5
        scales = 1
        codeformer_weight = 0
        codeformer_visibility = 0
        min_edge = 512
        if _output_aspect < 1:
            _resize_w = min_edge
            _resize_h = int(min_edge / _output_aspect)
        else:
            _resize_w = int(min_edge * _output_aspect)
            _resize_h = min_edge
        args = (1, scales, _resize_w, _resize_h, True, 'ESRGAN_4x', 'None',
                0, gfpgan_weight,
                codeformer_visibility, codeformer_weight)
        self.operator.devices.torch_gc()
        pp = self.operator.scripts_postprocessing.PostprocessedImage(cert_res.convert('RGB'))
        self.operator.scripts.scripts_postproc.run(pp, args)
        # pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
        self.operator.devices.torch_gc()

        return [pp.image]
