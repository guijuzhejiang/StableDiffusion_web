# coding=utf-8
# @Time : 2023/12/15 下午3:15
# @File : magic_cert.py
import copy
import os
import random
import string
import datetime

from utils.global_vars import CONFIG


class MagicUpscaler(object):
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
        _input_image_path = kwargs['input_image_paths'][0]
        _input_image_width, _input_image_height = _input_image.size

        # pattern = re.compile(r"user_(.*?)_history")
        # match = pattern.search(_input_image_path)
        _input_image_mode = os.path.basename(os.path.dirname(os.path.dirname(_input_image_path)))
        # if match:
        #     _input_image_mode = match.group(1)
        # else:
        #     _input_image_mode = 'facer'
        # hires
        # if _input_image_mode == 'avatar' or _input_image_mode == 'mirage':
        #     if self.operator.shared.sd_model.sd_checkpoint_info.model_name != 'dreamshaper_8':
        #         self.operator.shared.change_sd_model('dreamshaper_8')
        # elif _input_image_mode == 'facer':
        #     pass
        # else:
        #     if self.operator.shared.sd_model.sd_checkpoint_info.model_name != 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
        #         self.operator.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')

        if self.operator.update_progress(50):
            return {'success': True}

        gfpgan_weight = 0.5
        codeformer_visibility = 0
        codeformer_weight = 1

        scales = int(params['times'])

        args = (0, scales, None, None, True, 'ESRGAN_4x', 'None', 0, gfpgan_weight, codeformer_visibility,
                codeformer_weight)
        self.operator.devices.torch_gc()
        pp = self.operator.scripts_postprocessing.PostprocessedImage(_input_image.convert("RGB"))
        self.operator.scripts.scripts_postproc.run(pp, args)

        self.operator.devices.torch_gc()

        # celery_task.update_state(state='PROGRESS', meta={'progress': 80})
        if self.operator.update_progress(80):
            return {'success': True}

        dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], 'upscaler')

        os.makedirs(dir_path, exist_ok=True)

        img_fn = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{''.join([random.choice(string.ascii_letters) for c in range(6)])}.jpeg"
        img_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else '/service'}/user/image/fetch?imgpath={img_fn}"
        # pp.image.save(os.path.join(dir_path, img_fn), format="png", quality=100)
        pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=100, lossless=True)
        # celery_task.update_state(state='PROGRESS', meta={'progress': 90})
        if self.operator.update_progress(90):
            return {'success': True}
        return [pp.image]
