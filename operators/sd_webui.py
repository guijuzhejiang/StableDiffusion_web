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
import glob
import io
import importlib
import json
import os
import sys
import urllib.parse
import GPUtil
from PIL import Image
import datetime
import math
import random
import string
import traceback
import ujson
import cv2
import numpy as np

from guiju.magic_ai_conductor import MagicAiConductor
from lib.celery_workshop.operator import Operator
from utils.global_vars import CONFIG


class OperatorSD(Operator):
    """ stable diffusion """
    num = len(GPUtil.getGPUs())
    cache = True
    cuda = True
    enable = True
    celery_task_name = 'sd_task'

    model_name = {
        'avatar': 'dreamshaper_8',
        'other': 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting',
    }

    def __init__(self, gpu_idx=0):
        os.environ['ACCELERATE'] = 'True'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        print("use gpu:" + str(gpu_idx))
        super().__init__()
        # import lib
        self.extra_networks = importlib.import_module('modules.extra_networks')
        self.script_callbacks = importlib.import_module('modules.script_callbacks')
        self.dino = importlib.import_module('guiju.segment_anything_util.dino')
        self.dino_model_name = "GroundingDINO_SwinB (938MB)"
        self.initialize = getattr(importlib.import_module('lib.stable_diffusion.util'), 'initialize')
        self.shared = getattr(importlib.import_module('modules'), 'shared')
        self.ui_tempdir = getattr(importlib.import_module('modules'), 'ui_tempdir')
        self.sd_samplers = getattr(importlib.import_module('modules'), 'sd_samplers')
        self.config_states = getattr(importlib.import_module('modules'), 'config_states')
        self.modelloader = getattr(importlib.import_module('modules'), 'modelloader')
        self.extensions = getattr(importlib.import_module('modules'), 'extensions')
        self.extra_networks_hypernet = getattr(importlib.import_module('modules'), 'extra_networks_hypernet')
        self.scripts = getattr(importlib.import_module('modules'), 'scripts')
        self.sd_models = getattr(importlib.import_module('modules'), 'sd_models')
        self.sam = importlib.import_module('guiju.segment_anything_util.sam')
        self.sam_h = importlib.import_module('guiju.segment_anything_util.sam_h')
        self.predict_image = getattr(importlib.import_module('guiju.predictor_opennsfw2'), 'predict_image')
        self.logging = getattr(importlib.import_module('lib.common.common_util'), 'logging')
        self.logging = getattr(importlib.import_module('lib.common.common_util'), 'logging')
        self.img2img = getattr(importlib.import_module('modules'), 'img2img')
        self.txt2img = getattr(importlib.import_module('modules'), 'txt2img')
        self.devices = getattr(importlib.import_module('modules'), 'devices')
        self.scripts_postprocessing = getattr(importlib.import_module('modules'), 'scripts_postprocessing')

        self.insightface = importlib.import_module('insightface')
        self.swapper = self.insightface.model_zoo.get_model('models/insightface/models/inswapper_128.onnx')
        self.face_analysis = self.insightface.app.FaceAnalysis(name='buffalo_l', root='models/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

        self.faceid_predictor = getattr(importlib.import_module('guiju.faceid.faceid_predictor'), 'FaceIDPredictor')(self.face_analysis)

        self.shared.cmd_opts.listen = True
        self.shared.cmd_opts.debug_mode = True
        self.shared.cmd_opts.enable_insecure_extension_access = True
        self.shared.cmd_opts.xformers = True
        self.shared.cmd_opts.disable_tls_verify = True
        self.shared.cmd_opts.hide_ui_dir_config = True
        self.shared.cmd_opts.disable_tome = False
        self.shared.cmd_opts.lang = 'ch'
        self.shared.cmd_opts.disable_adetailer = False
        self.shared.cmd_opts.sd_checkpoint_cache = 0
        self.shared.cmd_opts.no_download_sd_model = True
        # self.shared.cmd_opts.ckpt = None

        # init
        self.initialize()
        self.sam.sam = self.sam.init_sam_model()
        # self.sam_h.sam = self.sam_h.init_sam_model()
        self.facer = getattr(importlib.import_module('guiju.facer_parsing.facer_parsing'), 'FaceParsing')()
        dino_model, dino_name = self.dino.load_dino_model2("GroundingDINO_SwinB (938MB)")
        self.dino.dino_model_cache[dino_name] = dino_model

        if self.shared.opts.clean_temp_dir_at_start:
            self.ui_tempdir.cleanup_tmpdr()
            print("cleanup temp dir")

        self.sd_samplers.set_samplers()

        self.extensions.list_extensions()

        config_state_file = self.shared.opts.restore_config_state_file
        self.shared.opts.restore_config_state_file = ""
        self.shared.opts.save(self.shared.config_filename)

        if os.path.isfile(config_state_file):
            print(f"*** About to restore extension state from file: {config_state_file}")
            with open(config_state_file, "r", encoding="utf-8") as f:
                config_state = json.load(f)
                self.config_states.restore_extension_config(config_state)
        elif config_state_file:
            print(f"!!! Config state backup not found: {config_state_file}")

        self.scripts.reload_scripts()
        print("load scripts")

        self.script_callbacks.model_loaded_callback(self.shared.sd_model)
        print("model loaded callback")

        self.modelloader.load_upscalers()

        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        print("reload script modules")

        self.sd_models.list_models()
        print("list SD models")

        self.shared.reload_hypernetworks()
        print("reload hypernetworks")

        self.extra_networks.initialize()
        self.extra_networks.register_extra_network(self.extra_networks_hypernet.ExtraNetworkHypernet())
        print("initialize extra networks")

        # init sam
        self.scripts.scripts_current = self.scripts.scripts_img2img
        self.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
        self.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

        self.cnet_idx = 3
        sam_idx = 4
        adetail_idx = 0
        tiled_diffusion_idx = 1
        tiled_vae_idx = 2
        self.scripts.scripts_img2img.alwayson_scripts[0], \
        self.scripts.scripts_img2img.alwayson_scripts[1], \
        self.scripts.scripts_img2img.alwayson_scripts[2], \
        self.scripts.scripts_img2img.alwayson_scripts[3], \
        self.scripts.scripts_img2img.alwayson_scripts[4], \
            = self.scripts.scripts_img2img.alwayson_scripts[sam_idx], \
              self.scripts.scripts_img2img.alwayson_scripts[tiled_diffusion_idx], \
              self.scripts.scripts_img2img.alwayson_scripts[tiled_vae_idx], \
              self.scripts.scripts_img2img.alwayson_scripts[self.cnet_idx], \
              self.scripts.scripts_img2img.alwayson_scripts[adetail_idx]

        self.scripts.scripts_txt2img.alwayson_scripts[0], \
        self.scripts.scripts_txt2img.alwayson_scripts[1], \
        self.scripts.scripts_txt2img.alwayson_scripts[2], \
        self.scripts.scripts_txt2img.alwayson_scripts[3], \
        self.scripts.scripts_txt2img.alwayson_scripts[4], \
            = self.scripts.scripts_txt2img.alwayson_scripts[sam_idx], \
              self.scripts.scripts_txt2img.alwayson_scripts[tiled_diffusion_idx], \
              self.scripts.scripts_txt2img.alwayson_scripts[tiled_vae_idx], \
              self.scripts.scripts_txt2img.alwayson_scripts[self.cnet_idx], \
              self.scripts.scripts_txt2img.alwayson_scripts[adetail_idx]

        # sam 24 args
        self.scripts.scripts_img2img.alwayson_scripts[0].args_from = 7
        self.scripts.scripts_img2img.alwayson_scripts[0].args_to = 31
        self.scripts.scripts_txt2img.alwayson_scripts[0].args_from = 7
        self.scripts.scripts_txt2img.alwayson_scripts[0].args_to = 31

        # tiled_diffusion 101 args
        self.scripts.scripts_img2img.alwayson_scripts[1].args_from = 31
        self.scripts.scripts_img2img.alwayson_scripts[1].args_to = 132
        self.scripts.scripts_txt2img.alwayson_scripts[1].args_from = 31
        self.scripts.scripts_txt2img.alwayson_scripts[1].args_to = 132

        # tiled_vae 7 args
        self.scripts.scripts_img2img.alwayson_scripts[2].args_from = 132
        self.scripts.scripts_img2img.alwayson_scripts[2].args_to = 139
        self.scripts.scripts_txt2img.alwayson_scripts[2].args_from = 132
        self.scripts.scripts_txt2img.alwayson_scripts[2].args_to = 139

        # controlnet 3 args
        self.scripts.scripts_img2img.alwayson_scripts[3].args_from = 4
        self.scripts.scripts_img2img.alwayson_scripts[3].args_to = 7
        self.scripts.scripts_txt2img.alwayson_scripts[3].args_from = 4
        self.scripts.scripts_txt2img.alwayson_scripts[3].args_to = 7

        # adetail 3 args
        self.scripts.scripts_img2img.alwayson_scripts[4].args_from = 1
        self.scripts.scripts_img2img.alwayson_scripts[4].args_to = 4
        self.scripts.scripts_txt2img.alwayson_scripts[4].args_from = 1
        self.scripts.scripts_txt2img.alwayson_scripts[4].args_to = 4

        # invisible detectmap
        self.shared.opts.control_net_no_detectmap = True

        # init lora
        self.extra_networks.initialize()
        self.extra_networks.register_default_extra_networks()
        self.script_callbacks.before_ui_callback()

        self.magic_conductor = MagicAiConductor(self)
        print('init done')

    def limit_and_compress_image(self, __cv_image, __output_height, quality=80):
        # 转pil
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        _, jpeg_data = cv2.imencode('.jpg', cv2.cvtColor(np.array(__cv_image), cv2.COLOR_RGBA2BGRA),
                                    encode_param)

        # 将压缩后的图像转换为PIL图像
        __cv_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')

        # limit height 768
        check_w, check_h = __cv_image.size
        print(f"before:{__cv_image.size}")

        if check_h > __output_height:
            tmp_w = int(__output_height * check_w / check_h)
            # tmp_w = int(tmp_w // 8 * 8)
            __cv_image = __cv_image.resize((tmp_w, __output_height))
            check_w, check_h = __cv_image.size

        tmp_h = math.ceil(check_h / 8) * 8
        tmp_w = math.ceil(check_w / 8) * 8
        left = int((tmp_w - check_w) / 2)
        right = tmp_w - check_w - left
        top = int((tmp_h - check_h) / 2)
        bottom = tmp_h - check_h - top
        __cv_image = cv2.copyMakeBorder(cv2.cvtColor(np.array(__cv_image), cv2.COLOR_RGBA2BGRA),
                                        top, bottom, left,
                                        right,
                                        cv2.BORDER_REPLICATE,
                                        value=(127, 127, 127))
        # # 压缩图像质量
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, jpeg_data = cv2.imencode('.jpg', np.array(__cv_image),
                                    encode_param)
        # 将压缩后的图像转换为PIL图像
        __cv_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')
        return __cv_image

    def __call__(self, *args, **kwargs):
        try:
            super().__call__(*args, **kwargs)

            # log start
            print(f"{str(datetime.datetime.now())} operation start !!!!!!!!!!!!!!!!!!!!!!!!!!")
            clean_args = {k: v for k, v in kwargs.items() if k != 'input_image'}
            clean_args['params'] = ujson.loads(kwargs['params'][0])
            print(clean_args)
            proceed_mode = kwargs['mode'][0]
            user_id = kwargs['user_id'][0]
            params = ujson.loads(kwargs['params'][0])
            origin = kwargs['origin']

            if 'imegaai' in origin:
                if 'www.' in origin:
                    client_origin = origin.replace('www', 'api')
                else:
                    client_protocol = origin.split('://')[0]
                    client_origin = origin.replace(f'{client_protocol}://', f'{client_protocol}://api.')
            else:
                client_origin = ''


            if self.update_progress(2):
                return {'success': True}

            # nsfw check
            if proceed_mode != 'wallpaper':
                if 'preset_index' in params.keys():
                    if self.predict_image(f"guiju/assets/preset/{proceed_mode}/{params['preset_index']}.jpg"):
                        return {'success': False, 'result': 'backend.check.error.nsfw'}
                else:
                    if self.predict_image(kwargs['input_image']):
                        return {'success': False, 'result': 'backend.check.error.nsfw'}

            # define task id
            pic_name = ''.join([random.choice(string.ascii_letters) for c in range(6)])

            # read input image
            if proceed_mode not in ['wallpaper', 'facer']:
                if 'preset_index' in params.keys() and params['preset_index'] is not None and params['preset_index'] >= 0:
                    _input_image = Image.open(f"guiju/assets/preset/{proceed_mode}/{params['preset_index']}.jpg")
                    _input_image_width, _input_image_height = _input_image.size

                else:
                    _input_image = Image.open(kwargs['input_image'])
                    _input_image_width, _input_image_height = _input_image.size

                # cache upload image
                _input_image.save(f"tmp/{proceed_mode}_origin_{pic_name}_save.png")

            else:
                _input_image_width, _input_image_height = 0, 0

            if self.update_progress(5):
                return {'success': True}

            # logging
            self.logging(
                f"[__call__][{datetime.datetime.now()}]:\n"
                f"[{pic_name}]:\n"
                f"{ujson.dumps(clean_args, indent=4)}",
                f"logs/sd_webui.log")

            if proceed_mode == 'facer':
                input_image_paths = [kwargs['input_image'], kwargs['input_image_tgt']]
            elif proceed_mode == 'hires':
                input_image_paths = [kwargs['input_image']]
            else:
                input_image_paths = None


            res = self.magic_conductor(proceed_mode,
                                       params=params,
                                       user_id=user_id,
                                       input_image=_input_image if proceed_mode not in ['wallpaper', 'facer'] else None,
                                       input_image_paths=input_image_paths,
                                       pic_name=pic_name)
            if isinstance(res, dict):
                return res

            # storage img
            img_urls = []
            dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], proceed_mode, user_id)
            # dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_{proceed_mode}_dir'], user_id)
            os.makedirs(dir_path, exist_ok=True)
            for res_idx, res_img in enumerate(res):
                img_save_path = os.path.join(dir_path, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
                res_img = res_img.convert("RGB")
                res_img.save(img_save_path, format="jpeg", quality=80, lossless=True)

                # cache output
                cache_fp = f"tmp/{proceed_mode}_{pic_name}_{res_idx}.jpg"
                res_img.save(cache_fp)

                # 限制缓存10张
                cache_list = sorted(os.listdir(dir_path))
                if len(cache_list) > 10:
                    os.remove(os.path.join(dir_path, cache_list[0]))
            else:
                for img_fn in sorted(os.listdir(dir_path), reverse=True):
                    # url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category={proceed_mode}"
                    url_fp = f"{'localhost:' + str(CONFIG['server']['port']) if CONFIG['local'] else f'{client_origin}/service'}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category={proceed_mode}"
                    img_urls.append(url_fp)
                if len(img_urls) < 10:
                    for i in range(10 - len(img_urls)):
                        img_urls.append('')

            # celery_task.update_state(state='PROGRESS', meta={'progress': 95})
            if self.update_progress(90):
                return {'success': True}
            else:
                # clear images
                for cache_img_fp in glob.glob(f'tmp/*{pic_name}*'):
                    if '_save' not in cache_img_fp:
                        os.remove(cache_img_fp)
                else:
                    return {'success': True, 'result': img_urls}

        except Exception:
            print('errrrr!!!!!!!!!!!!!!')
            self.logging(
                f"[predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        return {'success': False, 'result': 'backend.generate.error.failed'}
