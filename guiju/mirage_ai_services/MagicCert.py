# coding=utf-8
# @Time : 2023/12/15 下午3:15
# @File : magic_cert.py
import os
import datetime

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
        # _output_aspect = float(params['aspect'])

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
            # padding = 0
            # resize_mask = sam_person_result[1].resize((_input_image_width - padding, _input_image_height - padding))
            # padding_mask = Image.new("RGB", _input_image.size, (0, 0, 0, 1))
            # padding_mask.paste(resize_mask, (int(padding / 2), int(padding / 2)))
            # padding_mask = padding_mask.convert('1')
            #
            # padding_sam = Image.new('RGBA', _input_image.size)
            # padding_sam.paste(sam_person_result[2], (0, 0), padding_mask)
            # sam_person_result[2] = padding_sam

        if self.operator.update_progress(60):
            return {'success': True}

        # new canvas
        cert_res = Image.new("RGBA", sam_person_result[2].size, _bg_color)
        # foreground = sam_person_result[2].crop(face_box)
        cert_res.paste(sam_person_result[2], (0, 0), mask=sam_person_result[2])

        # storage img
        dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], 'cert')

        os.makedirs(dir_path, exist_ok=True)
        img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
        output_w, output_h = sam_person_result[2].size
        # face fix
        gfpgan_weight = 0.5
        scales = 1
        codeformer_weight = 0
        codeformer_visibility = 0
        args = (1, scales, output_w, output_h, True, 'ESRGAN_4x', 'None',
                0, gfpgan_weight,
                codeformer_visibility, codeformer_weight)
        self.operator.devices.torch_gc()
        pp = self.operator.scripts_postprocessing.PostprocessedImage(cert_res.convert('RGB'))
        self.operator.scripts.scripts_postproc.run(pp, args)
        # pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
        self.operator.devices.torch_gc()

        return [pp.image]
