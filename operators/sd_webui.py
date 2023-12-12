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
import re
import sys
import urllib.parse
from collections import OrderedDict

import GPUtil
from PIL import Image, ImageDraw
import copy
import datetime
import math
import random
import string
import traceback
import ujson
import cv2
import numpy as np

from lora_config import lora_model_dict, lora_gender_dict, lora_model_common_dict, lora_place_dict, lora_bg_common_dict, \
    lora_haircut_common_dict, lora_haircut_male_dict, lora_haircut_female_dict, lora_hair_color_dict, \
    male_avatar_reference_dict, female_avatar_reference_dict, reference_dir, lora_avatar_dict, lora_mirage_dict
from lib.celery_workshop.operator import Operator
from utils.global_vars import CONFIG

samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
    ('Euler', 'sample_euler', ['k_euler'], {}),
    ('LMS', 'sample_lms', ['k_lms'], {}),
    ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'discard_next_to_last_sigma': True}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'discard_next_to_last_sigma': True, "uses_ensd": True}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {"uses_ensd": True, "second_order": True}),
    ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {}),
    ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {"second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {"brownian_noise": True}),
    ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
    ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
    ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'],
     {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'],
     {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a_ka'],
     {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    ('DPM++ 2M Karras', 'sample_dpmpp_2m', ['k_dpmpp_2m_ka'], {'scheduler': 'karras'}),
    ('DPM++ SDE Karras', 'sample_dpmpp_sde', ['k_dpmpp_sde_ka'],
     {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'],
     {'scheduler': 'karras', "brownian_noise": True}),
]


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
        self.face_analysis = self.insightface.app.FaceAnalysis(name='buffalo_l', root='models/insightface')
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

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

        print('init done')

    def configure_image(self, image, person_pos, min_edge=512, quality=90, color=(255, 255, 255)):
        person_pos = [int(x) for x in person_pos]
        # 将PIL RGBA图像转换为BGR图像
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

        # 获取原始图像的尺寸
        original_height, original_width = cv_image.shape[:2]

        # 计算模特图像的长宽比
        person_height = person_pos[3] - person_pos[1]
        person_width = person_pos[2] - person_pos[0]
        person_ratio = person_width / person_height

        # 裁剪或扩展
        # left && top
        cropped_left = person_pos[0] if person_pos[0] > 0 else 0
        cropped_top = person_pos[1] if person_pos[1] > 0 else 0
        cropped_right = person_pos[2] if person_pos[2] <= original_width else original_width
        cropped_bottom = person_pos[3] if person_pos[3] <= original_height else original_height
        cv_image = cv_image[cropped_top:cropped_bottom, cropped_left:cropped_right]

        cv_image = cv2.copyMakeBorder(cv_image,
                                      abs(person_pos[1]) if person_pos[1] < 0 else 0,
                                      person_pos[3] - original_height if person_pos[3] > original_height else 0,
                                      abs(person_pos[0]) if person_pos[0] < 0 else 0,
                                      person_pos[2] - original_width if person_pos[2] > original_width else 0,
                                      cv2.BORDER_CONSTANT, value=color)

        cur_height, cur_width = cv_image.shape[:2]
        # cur_ratio = cur_width / cur_height

        # # 计算应该添加的填充量
        # if cur_ratio > target_ratio:
        #     # 需要添加垂直box
        #     target_height = int(cur_width / target_ratio)
        #
        #     top = int((target_height - cur_height) / 2)
        #     bottom = target_height - cur_height - top
        #     padded_image = cv2.copyMakeBorder(cv_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
        # else:
        #     # 需要添加水平box
        #     target_width = int(person_height * target_ratio)
        #
        #     left = int((target_width - cur_width) / 2)
        #     right = target_width - cur_width - left
        #     padded_image = cv2.copyMakeBorder(cv_image, 0, 0, left, right, cv2.BORDER_CONSTANT,
        #                                       value=color)

        # padded_image = cv2.cvtColor(np.array(padded_image), cv2.COLOR_BGRA2RGBA)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        _, jpeg_data = cv2.imencode('.jpg', cv_image,
                                    encode_param)

        # 将压缩后的图像转换为PIL图像
        __pil_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')

        __pil_w, __pil_h = __pil_image.size
        min_edge = min(__pil_w, __pil_h)
        min_index = [__pil_w, __pil_h].index(min_edge)
        if min_index == 0:
            res_image = __pil_image.resize((512, int(__pil_h / __pil_w * 512)))
        else:
            res_image = __pil_image.resize((int(__pil_w / __pil_h * 512), 512))
        return res_image

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

    def get_prompt(self, _age, _viewpoint, _model_type, _place_type, _model_mode=0):
        sd_positive_common_prompts = [
            '(best quality:1.2)',
            '(high quality:1.2)',
            'high details',
            '(Realism:1.4)',
            'masterpiece',
            'extremely detailed,extremely delicate,Amazing,8k quality,',
        ]
        sd_positive_model_prompts_dict = OrderedDict({
            'age': [
                # child
                f'(child:1.3)',
                # youth
                # f'(youth:1.3){"" if _gender else ",<lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>"}',
                f'(youth:1.3),20 years old',
                # middlescent
                '(middlescent:1.3)',
            ],
            'common': [
                'full body',
                # 'a person',
                'correct body proportions,good figure',
                # 'detailed fingers',
                'realistic fingers',
                # 'detailed hand',
                'realistic hand',
                # 'detailed foot',
                'realistic body',
                # '(out of frame:1.3)',
                '' if _viewpoint == 2 else 'posing for a photo,realistic face',
                'footwear',
                'tall',
                # 'Fixhand',
                # 'hand101',
                '(simple background:1.3)',
                # '(plain background:1.3)',
                # 'natural skin texture',
                # 'beautiful fingers',
                # 'clear fingernails',
                # 'realistic hand appearance',
                # 'elegant hand gesture',
                # 'realistic hand anatomy',
            ],
            'viewpoint': [
                # 正面
                'light smile,looking at viewer,beautiful detailed face,beautiful detailed nose,beautiful detailed eyes',
                # 侧面
                f'{"" if lora_model_dict[_model_type]["gender"] == 1 else "<lora:sideface_v1.0:0.6>,sideface,"}facing to the side,a side portrait photo of a people,(looking to the side:1.5)',
                # 反面
                '(a person with their back to the camera:1.5)'
            ]
        })

        # model prompt
        sd_positive_model_prompts_dict['common'] = [x for x in sd_positive_model_prompts_dict['common'] if x]
        sd_positive_model_prompts_dict['age'] = [sd_positive_model_prompts_dict['age'][_age]]
        sd_positive_model_prompts_dict['viewpoint'] = [sd_positive_model_prompts_dict['viewpoint'][_viewpoint]]

        sd_model_positive_prompt = ','.join(sd_positive_common_prompts)
        if _viewpoint == 2:
            sd_model_positive_prompt += ','.join(
                [x for i in [sd_positive_model_prompts_dict['age'], sd_positive_model_prompts_dict['viewpoint']] for x
                 in i])
        else:
            sd_model_positive_prompt += ','.join([i for x in sd_positive_model_prompts_dict.values() for i in x])

        # model negative
        sd_model_negative_prompt = f'(NSFW:1.8),(pussy:1.8),(vagina:1.8),(sexual organ:1.8),mutated hands and fingers,malformed hands,(masks:1.5),text,logo,tattoos,abstraction,distortion,(barefoot:1.3),(extra clothes:1.5),(clothes:1.5),paintings,sketches,(worst quality:2),(low quality:2),(normal quality:2),clothing,pants,shorts,t-shirt,dress,sleeves,lowres,((monochrome)),((grayscale)),duplicate,morbid,mutilated,poorly drawn face,skin spots,acnes,skin blemishes,age spot,glans,extra fingers,fewer fingers,bad anatomy,bad hands,error,missing fingers,missing arms,missing legs,extra digit,fewer digits,cropped,worst quality,blurry,poorly drawn hands,mutation,deformed,extra limbs,extra arms,extra legs,malformed limbs,too many fingers,long neck,cross-eyed,polar lowres,bad body,gross proportions,fused fingers,bad proportion body to legs,mirrored image,mirrored noise,(bad_prompt_version2:0.8),aged up,old fingers,bad feet,wrong feet bottom render,wrong toes,extra toes,missing toes,weird toes,2 upper,2 lower,2 head,3 hand,3 feet,3 legs,3 arms'

        # lora
        if _viewpoint == 2:
            lora_prompt_list = []
            ad_face_positive_prompt = ''

        else:
            lora_model = lora_model_dict[_model_type]
            lora_prompt_list = [f"<lora:{lora_model['lora_name']}:{lora_model['weight']}>",
                                lora_gender_dict[lora_model['gender']],
                                ]
            if len(lora_model['prompt']) > 0:
                lora_prompt_list.append(lora_model['prompt'])

            ad_face_positive_prompt = ','.join(lora_prompt_list)
            ad_face_positive_prompt += ',' + ','.join(
                [f"<lora:{lora_model_common_dict[0]['lora_name']}:{lora_model_common_dict[0]['weight']}>"])

        for lora_common in lora_model_common_dict:
            lora_prompt_list.append(f"<lora:{lora_common['lora_name']}:{lora_common['weight']}>")

        sd_model_positive_prompt = f"{','.join(lora_prompt_list)},{sd_model_positive_prompt}"

        print(f'sd_model_positive_prompt: {sd_model_positive_prompt}')
        print(f'sd_model_negative_prompt: {sd_model_negative_prompt}')
        print(f'ad_face_positive_prompt: {ad_face_positive_prompt}')

        # bg prompt
        bg_prmpt_list = [lora_place_dict[_place_type]['prompt'],
                         ','.join([f"<lora:{bg_common['lora_name']}:{bg_common['weight']}>" for bg_common in
                                   lora_bg_common_dict]),
                         ','.join(sd_positive_common_prompts),
                         '(no humans:1.3),8k uhd,dramatic scene,Epic composition,raw photo,huge_filesize,highres,magazine cover,high saturation,poster']
        sd_bg_positive_prompt = ','.join(bg_prmpt_list)
        sd_bg_negative_prompt = '(NSFW:1.8),(person:1.4),(hands),(feet),(shoes),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(overexposure:1.5),(exposure:1.5),paintings,sketches,(worst quality:2),(low quality:2),(normal quality:2),clothing,pants,shorts,t-shirt,dress,sleeves,lowres,((monochrome)),((grayscale)),duplicate,morbid,error,cropped,worst quality,blurry,deformed,mirrored image,mirrored noise,polar lowres'
        # 3 feet,extra long leg,super long leg,wrong feet bottom render

        print(f'sd_bg_positive_prompt: {sd_bg_positive_prompt}')
        print(f'sd_bg_negative_prompt: {sd_bg_negative_prompt}')

        return sd_model_positive_prompt, sd_model_negative_prompt, ad_face_positive_prompt, sd_bg_positive_prompt, sd_bg_negative_prompt

    def proceed_human_transform(self, _params, _task_type, _batch_size, _init_img):
        prompt_styles = None
        sketch = None
        init_img_with_mask = None
        inpaint_color_sketch = None
        inpaint_color_sketch_orig = None
        init_img_inpaint = None
        init_mask_inpaint = None
        mask_alpha = 0
        restore_faces = False
        tiling = False
        n_iter = 1
        image_cfg_scale = 1.5
        seed = -1.0
        subseed = -1.0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        selected_scale_tab = 0
        scale_by = 1
        inpaint_full_res_padding = 32
        img2img_batch_input_dir = ''
        img2img_batch_output_dir = ''
        img2img_batch_inpaint_mask_dir = ''
        override_settings_texts = []
        # controlnet args
        controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[self.cnet_idx].get_default_ui_unit()
        controlnet_args_unit1.enabled = False
        controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit2.enabled = False
        controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit3.enabled = False

        result_images = []
        pic_name = ''.join([random.choice(string.ascii_letters) for c in range(6)])

        if _task_type == 'gender':
            # segment
            sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "person", 0.4, _init_img)

            if len(sam_result) == 0:
                # return {'success': False, 'result': '未检测到人体'}
                return {'success': False, 'result': 'backend.magic-mirror.error.no-body'}

            else:
                sam_result_tmp_png_fp = []
                for idx, sam_mask_img in enumerate(sam_result):
                    cache_fp = f"tmp/mirror_age_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                    sam_mask_img.save(cache_fp)
                    sam_result_tmp_png_fp.append({'name': cache_fp})
            # 性別
            gender_prompt = ['GS-Girlish,GS-DeMasculate', 'GS-Womanly,GS-DeMasculate'] if _params[_task_type][
                                                                                              'gender'] == 'female' \
                else ['GS-Boyish,GS-DeFeminize', 'GS-Masculine,GS-DeFeminize']
            gender_prompt = gender_prompt[0] if _params[_task_type]['is_elder'] == 'young' else gender_prompt[1]

            denoising_strength_min = 0.05
            denoising_strength_max = 0.5
            denoising_strength = (1 - _params[_task_type]['sim']) * (
                    denoising_strength_max - denoising_strength_min) + denoising_strength_min
            steps = 20
            sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
            cfg_scale = 10
            resize_mode = 2  # resize and fill
            mask_blur = 0
            inpainting_mask_invert = 0  # inpaint masked
            inpaint_full_res = 1  # choices=["Whole picture", "Only masked"]
            inpainting_fill = 1  # masked content original

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            sd_positive_prompt = f'{gender_prompt},<lora:polyhedron_new_skin_v1.1:0.1>,(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo,realistic,character close-up'
            sd_negative_prompt = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(humans:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions'

            print("-------------------gender logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"dino_prompt: person")
            print(f"denoising_strength: {denoising_strength}")
            print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

            # adetail
            adetail_enabled = True
            face_args = {'ad_model': 'face_yolov8n.pt',
                         'ad_prompt': f'',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            hand_args = {'ad_model': 'None',
                         'ad_prompt': '',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                        # sam
                        True, False, 0, _init_img,
                        sam_result_tmp_png_fp,
                        0,  # sam_output_chosen_mask
                        False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

        elif _task_type == 'age':
            # segment
            sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "person", 0.45, _init_img)
            if len(sam_result) == 0:
                # return {'success': False, 'result': '未检测到人体'}
                return {'success': False, 'result': 'backend.magic-mirror.error.no-body'}

            else:
                sam_result_tmp_png_fp = []
                for idx, sam_mask_img in enumerate(sam_result):
                    cache_fp = f"tmp/mirror_age_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                    sam_mask_img.save(cache_fp)
                    sam_result_tmp_png_fp.append({'name': cache_fp})

            # age
            denoising_strength_min = 0.05
            denoising_strength_max = 0.4
            denoising_strength = (1 - _params[_task_type]['sim']) * (
                    denoising_strength_max - denoising_strength_min) + denoising_strength_min
            steps = 20
            sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
            cfg_scale = 10
            resize_mode = 0  # just resize
            mask_blur = 0
            inpainting_mask_invert = 0  # inpaint masked
            inpaint_full_res = 1  # choices=["Whole picture", "Only masked"]
            inpainting_fill = 1  # masked content: original

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            sd_positive_prompt = f'<lora:age_slider_v20:{str(_params[_task_type]["weight"] * 10 - 5)}>,<lora:polyhedron_new_skin_v1.1:0.1>,photographic reality,character close-up,photorealistic,realistic,(best quality:1.2),(high quality:1.2),high details'
            sd_negative_prompt = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(humans:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions'

            print("-------------------age logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"dino_prompt: person")
            print(f"denoising_strength: {denoising_strength}")
            print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

            # adetail
            adetail_enabled = True
            face_args = {'ad_model': 'face_yolov8n.pt',
                         'ad_prompt': f'',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            hand_args = {'ad_model': 'None',
                         'ad_prompt': '',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                        # sam
                        True, False, 0, _init_img,
                        sam_result_tmp_png_fp,
                        0,  # sam_output_chosen_mask
                        False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

        elif _task_type == 'face_expression':
            # segment
            # sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "face.glasses",
            #                                                 0.3, _init_img)
            sam_result = self.facer(_init_img, keep='face')
            if sam_result is None:
                # return {'success': False, 'result': '未检测到人脸'}
                return {'success': False, 'result': 'backend.magic-mirror.error.no-face'}

            else:
                sam_result_tmp_png_fp = []
                for idx in range(3):
                    cache_fp = f"tmp/face_expression_{idx}_{pic_name}{'_save' if idx == 1 else ''}.png"
                    if idx == 1:
                        sam_result.save(cache_fp, format='PNG')
                    else:
                        _init_img.save(cache_fp, format='PNG')
                    sam_result_tmp_png_fp.append({'name': cache_fp})
                # for idx, sam_mask_img in enumerate(sam_result):
                #     cache_fp = f"tmp/face_expression_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                #     sam_mask_img.save(cache_fp)
                #     sam_result_tmp_png_fp.append({'name': cache_fp})
            # face_expression
            denoising_strength_min = 0.1
            denoising_strength_max = 0.4
            denoising_strength = (1 - _params[_task_type]['sim']) * (
                    denoising_strength_max - denoising_strength_min) + denoising_strength_min
            steps = 20
            sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
            cfg_scale = 10
            resize_mode = 2  # resize and fill
            mask_blur = 0
            inpainting_mask_invert = 0  # inpaint masked
            inpaint_full_res = 1  # choices=["Whole picture", "Only masked"]
            inpainting_fill = 1  # masked content: original

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            sd_positive_prompt = f'({str(_params[_task_type]["expression"])}512:1.3),<lora:polyhedron_new_skin_v1.1:0.1>,(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo,realistic,character close-up'
            sd_negative_prompt = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(humans:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions'

            print("-------------------face_expression logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"dino_prompt: face.glasses")
            print(f"denoising_strength: {denoising_strength}")
            print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

            # adetail
            adetail_enabled = False
            face_args = {'ad_model': 'face_yolov8n.pt',
                         'ad_prompt': f'',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            hand_args = {'ad_model': 'None',
                         'ad_prompt': '',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                        # sam
                        True, False, 0, _init_img,
                        sam_result_tmp_png_fp,
                        0,  # sam_output_chosen_mask
                        False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

        elif _task_type == 'eye_size':
            # segment
            sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "face.glasses",
                                                            0.5, _init_img)
            if len(sam_result) == 0:
                return {'success': False, 'result': '未检测到人脸'}
            else:
                sam_result_tmp_png_fp = []
                for idx, sam_mask_img in enumerate(sam_result):
                    cache_fp = f"tmp/eye_size_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                    sam_mask_img.save(cache_fp)
                    sam_result_tmp_png_fp.append({'name': cache_fp})
            # eye_size
            denoising_strength = 0.35
            steps = 20
            sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
            cfg_scale = 10
            resize_mode = 0  # just resize
            mask_blur = 0
            inpainting_mask_invert = 0  # inpaint masked
            inpaint_full_res = 1  # choices=["Whole picture", "Only masked"]
            inpainting_fill = 1  # masked content: original

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            sd_positive_prompt = f"<lora:eye_size_slider_v1:{str(_params[_task_type]['weight'] * 16 - 8)}>,<lora:polyhedron_new_skin_v1.1:0.1>,(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo,realistic,character close-up"
            sd_negative_prompt = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(humans:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions'

            print("-------------------eye_size logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"dino_prompt: face.glasses")
            print(f"denoising_strength: {denoising_strength}")
            print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

            # adetail
            adetail_enabled = False
            face_args = {'ad_model': 'face_yolov8n.pt',
                         'ad_prompt': f'',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            hand_args = {'ad_model': 'None',
                         'ad_prompt': '',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                        # sam
                        True, False, 0, _init_img,
                        sam_result_tmp_png_fp,
                        0,  # sam_output_chosen_mask
                        False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

        elif _task_type == 'curly_hair':
            # segment
            # sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "hair",
            #                                                 0.36, _init_img)
            sam_result = self.facer(_init_img, keep='hair')
            if sam_result is None:
                # return {'success': False, 'result': '未检测到头发'}
                return {'success': False, 'result': 'backend.magic-mirror.error.no-hair'}

            else:
                sam_result_tmp_png_fp = []
                # for idx, sam_mask_img in enumerate(sam_result):
                #     cache_fp = f"tmp/curly_hair_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                #     sam_mask_img.save(cache_fp)
                #     sam_result_tmp_png_fp.append({'name': cache_fp})
                for idx in range(3):
                    cache_fp = f"tmp/curly_hair_{idx}_{pic_name}{'_save' if idx == 1 else ''}.png"
                    if idx == 1:
                        sam_result.save(cache_fp, format='PNG')
                    else:
                        _init_img.save(cache_fp, format='PNG')
                    sam_result_tmp_png_fp.append({'name': cache_fp})

            # curly_hair
            denoising_strength = 0.8
            steps = 20
            sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
            cfg_scale = 10
            resize_mode = 0  # just resize
            mask_blur = 0
            inpainting_mask_invert = 0  # inpaint masked
            inpaint_full_res = 1  # choices=["Whole picture", "Only masked"]
            inpainting_fill = 1  # masked content: original

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            sd_positive_prompt = f"<lora:curly_hair_slider_v1:{str(_params[_task_type]['weight'] * 16 - 8)}>,fluffy hair,lush hair,{'straight hair,' if int(_params[_task_type]['weight']) == 0 else ''}{'curly hair,' if int(_params[_task_type]['weight']) == 1 else ''}<lora:polyhedron_new_skin_v1.1:0.1>,(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo, realistic,character close-up"
            sd_negative_prompt = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(humans:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions'

            print("-------------------curly_hair logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"dino_prompt: hair")
            print(f"denoising_strength: {denoising_strength}")
            print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

            # adetail
            adetail_enabled = False
            face_args = {'ad_model': 'face_yolov8n.pt',
                         'ad_prompt': f'',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            hand_args = {'ad_model': 'None',
                         'ad_prompt': '',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                        # sam
                        True, False, 0, _init_img,
                        sam_result_tmp_png_fp,
                        0,  # sam_output_chosen_mask
                        False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

        elif _task_type == 'muscle':
            # segment
            sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "breasts.arms.legs.abdomen", 0.31,
                                                            _init_img)
            if len(sam_result) == 0:
                # return {'success': False, 'result': '未检测到人体'}
                return {'success': False, 'result': 'backend.magic-mirror.error.no-body'}

            else:
                sam_result_tmp_png_fp = []
                for idx, sam_mask_img in enumerate(sam_result):
                    cache_fp = f"tmp/muscle_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                    sam_mask_img.save(cache_fp)
                    sam_result_tmp_png_fp.append({'name': cache_fp})
            # muscle
            denoising_strength = 0.5
            steps = 20
            sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
            cfg_scale = 10
            resize_mode = 0  # just resize
            mask_blur = 0
            inpainting_mask_invert = 0  # inpaint masked
            inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
            inpainting_fill = 1  # masked content: original

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            sd_positive_prompt = f"<lora:muscle_slider_v1:{str(_params[_task_type]['weight'] * 8 - 3)}>,<lora:polyhedron_new_skin_v1.1:0.1>,(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo,realistic,character close-up"
            sd_negative_prompt = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(humans:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions'

            print("-------------------muscle logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"dino_prompt: breasts.arms.legs.abdomen")
            print(f"denoising_strength: {denoising_strength}")
            print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

            # adetail
            adetail_enabled = False
            face_args = {'ad_model': 'face_yolov8n.pt',
                         'ad_prompt': f'',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            hand_args = {'ad_model': 'None',
                         'ad_prompt': '',
                         'ad_negative_prompt': '',
                         'ad_confidence': 0.3,
                         'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                         'ad_y_offset': 0,
                         'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                         'ad_denoising_strength': 0.4,
                         'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                         'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                         'ad_inpaint_height': 512,
                         'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                         'ad_cfg_scale': 7,
                         'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                         'ad_restore_face': False,
                         'ad_controlnet_model': 'None',
                         'ad_controlnet_module': 'inpaint_global_harmonious',
                         'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                         'ad_controlnet_guidance_end': 1,
                         'is_api': ()}
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,
                        # controlnet args
                        True, False, 0, _init_img,
                        sam_result_tmp_png_fp,
                        0,  # sam_output_chosen_mask
                        False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

        _input_image_width, _input_image_height = _init_img.size
        res = self.img2img.img2img(task_id,
                                   4,
                                   sd_positive_prompt,
                                   sd_negative_prompt,
                                   prompt_styles, _init_img,
                                   sketch,
                                   init_img_with_mask, inpaint_color_sketch,
                                   inpaint_color_sketch_orig,
                                   init_img_inpaint, init_mask_inpaint,
                                   steps, sampler_index, mask_blur, mask_alpha,
                                   inpainting_fill,
                                   restore_faces,
                                   tiling,
                                   n_iter,
                                   _batch_size,  # batch_size
                                   cfg_scale, image_cfg_scale,
                                   denoising_strength, seed,
                                   subseed,
                                   subseed_strength, seed_resize_from_h,
                                   seed_resize_from_w,
                                   seed_enable_extras,
                                   selected_scale_tab, _input_image_height,
                                   _input_image_width,
                                   scale_by,
                                   resize_mode,
                                   inpaint_full_res,
                                   inpaint_full_res_padding, inpainting_mask_invert,
                                   img2img_batch_input_dir,
                                   img2img_batch_output_dir,
                                   img2img_batch_inpaint_mask_dir,
                                   override_settings_texts,
                                   *sam_args)

        self.devices.torch_gc()

        # clear images
        for cache_img_fp in glob.glob(f'tmp/*{pic_name}*'):
            if '_save' not in cache_img_fp:
                os.remove(cache_img_fp)

        return res

    def proceed_hair(self, _selected_index, _task_type, _batch_size, _init_img, _pic_name, return_list=True,
                     gender='male'):
        uid_name = ''.join([random.choice(string.ascii_letters) for c in range(4)])

        # common
        prompt_styles = None
        sketch = None
        init_img_with_mask = None
        inpaint_color_sketch = None
        inpaint_color_sketch_orig = None
        init_img_inpaint = None
        init_mask_inpaint = None
        mask_alpha = 0
        restore_faces = False
        tiling = False
        n_iter = 1
        image_cfg_scale = 1.5
        seed = -1.0
        subseed = -1.0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        selected_scale_tab = 0
        scale_by = 1
        inpaint_full_res_padding = 32
        img2img_batch_input_dir = ''
        img2img_batch_output_dir = ''
        img2img_batch_inpaint_mask_dir = ''
        override_settings_texts = []
        # controlnet args
        controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[self.cnet_idx].get_default_ui_unit()
        controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit2.enabled = False
        controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit3.enabled = False

        adetail_enabled = False
        face_args = {'ad_model': 'face_yolov8n.pt',
                     'ad_prompt': f'',
                     'ad_negative_prompt': '',
                     'ad_confidence': 0.3,
                     'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                     'ad_y_offset': 0,
                     'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                     'ad_denoising_strength': 0.4,
                     'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                     'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                     'ad_inpaint_height': 512,
                     'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                     'ad_cfg_scale': 7,
                     'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                     'ad_restore_face': False,
                     'ad_controlnet_model': 'None',
                     'ad_controlnet_module': 'inpaint_global_harmonious',
                     'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                     'ad_controlnet_guidance_end': 1,
                     'is_api': ()}
        hand_args = {'ad_model': 'None',
                     'ad_prompt': '',
                     'ad_negative_prompt': '',
                     'ad_confidence': 0.3,
                     'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0,
                     'ad_y_offset': 0,
                     'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                     'ad_denoising_strength': 0.4,
                     'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                     'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                     'ad_inpaint_height': 512,
                     'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False,
                     'ad_cfg_scale': 7,
                     'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1,
                     'ad_restore_face': False,
                     'ad_controlnet_model': 'None',
                     'ad_controlnet_module': 'inpaint_global_harmonious',
                     'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                     'ad_controlnet_guidance_end': 1,
                     'is_api': ()}

        cfg_scale = 10
        mask_blur = 0
        resize_mode = 1  # just resize
        sampler_index = 15
        inpaint_full_res = 0 if _task_type == 'haircut' else 1  # choices=["Whole picture", "Only masked"]
        inpainting_fill = 1  # masked content original
        denoising_strength = 0.75 if _task_type == 'haircut' else 0.5
        steps = 20

        if _task_type == 'haircut':
            # 切割face.glasses
            sam_result = self.facer(_init_img, keep='face')
            # sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, 'face.neck',
            #                                                   0.3,
            #                                                   _init_img.convert('RGBA'))
            # _w, _h = sam_result[1].size
            # padding = 4
            # resize_mask = sam_result[1].resize((_w - padding, _h - padding))
            # padding_mask = Image.new("RGB", sam_result[1].size, (0, 0, 0, 1))
            # padding_mask.paste(resize_mask, (int(padding / 2), int(padding / 2)))
            # person_result = padding_mask.convert('L')
            #
            # sam_hair_result, _ = self.sam.sam_predict(self.dino_model_name, 'hair',
            #                                                 0.3,
            #                                                 _init_img.convert('RGBA'))
            # # person_result = sam_result[1].convert('L')
            # sam_hair_result = sam_hair_result[1].convert('L')
            #
            # # result_array = np.bitwise_xor(np.array(person_result), np.array(sam_hair_result))
            # result_array = np.maximum(np.array(person_result) - np.array(sam_hair_result), 0)
            # np.bitwise_and(array1, array2)
            # 将结果数组转换为 PIL 图像
            # sam_result = Image.fromarray(result_array)

        else:
            # 切割hair
            # sam_result, person_boxes = self.sam_h.sam_predict(self.dino_model_name, 'hair',
            #                                                   0.4,
            #                                                   _init_img.convert('RGBA'))
            sam_result = self.facer(_init_img, keep='hair')
        # _init_img = sam_result[2].convert('RGBA')

        sam_result_tmp_png_fp = []
        if sam_result is not None:
            for idx in range(3):
                cache_fp = f"tmp/hair_{_task_type}_{idx}_{uid_name}_{_pic_name}{'_save' if idx == 1 else ''}.png"
                if idx == 1:
                    sam_result.save(cache_fp, format='PNG')
                else:
                    _init_img.save(cache_fp, format='PNG')
                sam_result_tmp_png_fp.append({'name': cache_fp})
            # else:
            #     sam_result_tmp_png_fp[0] = sam_result_tmp_png_fp[-1]

        else:
            # return {'success': False, 'result': f'未切割到{"人脸" if _task_type=="haircut" else "头发"}'}
            return {'success': False,
                    'result': f'backend.magic-mirror.error.no-{"face" if _task_type == "haircut" else "hair"}'}

        # img2img
        inpainting_mask_invert = 1 if _task_type == 'haircut' else 0  # 0: inpaint masked 1: inpaint not masked
        task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

        if _task_type == 'haircut':
            sd_positive_prompt = ','.join([lora_haircut_male_dict[_selected_index]['prompt'] if gender == 'male' else
                                           lora_haircut_female_dict[_selected_index]['prompt'],
                                           lora_haircut_common_dict['positive_prompt']])
            sd_negative_prompt = lora_haircut_common_dict['negative_prompt']
        else:
            # sd_positive_prompt = lora_hair_color_dict[_selected_index]['prompt']
            sd_positive_prompt = ','.join([lora_hair_color_dict[_selected_index]['prompt'],
                                           '(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo,realistic,'])
            sd_negative_prompt = lora_haircut_common_dict['negative_prompt']

            controlnet_args_unit1.enabled = True
            controlnet_args_unit1.batch_images = ''
            controlnet_args_unit1.control_mode = 'Balanced'
            controlnet_args_unit1.guidance_end = 1
            controlnet_args_unit1.guidance_start = 0  # ending control step
            controlnet_args_unit1.image = None
            controlnet_args_unit1.low_vram = False
            controlnet_args_unit1.model = 't2iadapter_canny_sd14v1'
            controlnet_args_unit1.module = 'canny'
            controlnet_args_unit1.pixel_perfect = True
            controlnet_args_unit1.resize_mode = 'Resize and Fill'
            controlnet_args_unit1.processor_res = 512
            controlnet_args_unit1.threshold_a = 64
            controlnet_args_unit1.threshold_b = 64
            controlnet_args_unit1.weight = 1

        print(f"-------------------{_task_type} logger-----------------")
        print(f"sd_positive_prompt: {sd_positive_prompt}")
        print(f"sd_negative_prompt: {sd_negative_prompt}")
        print(f"dino_prompt: {'face.glasses' if _task_type == 'haircut' else 'hair'}")
        print(f"denoising_strength: {denoising_strength}")
        print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")

        # 不参考原图，参考分割图
        # _init_img = sam_result[2]

        sam_args = [0,
                    adetail_enabled, face_args, hand_args,  # adetail args
                    controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,
                    # controlnet args
                    True, False, 0, _init_img,
                    sam_result_tmp_png_fp,
                    0,  # sam_output_chosen_mask
                    False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                    '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                    True, True, '',
                    # tiled diffsuion
                    False, 'MultiDiffusion', False, True,
                    1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                    64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                    '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                    '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                    # tiled_vae
                    False, 256, 48, True, True, True,
                    False
                    ]

        _input_image_width, _input_image_height = _init_img.size
        res = self.img2img.img2img(task_id,
                                   4,
                                   sd_positive_prompt,
                                   sd_negative_prompt,
                                   prompt_styles, _init_img,
                                   sketch,
                                   init_img_with_mask, inpaint_color_sketch,
                                   inpaint_color_sketch_orig,
                                   init_img_inpaint, init_mask_inpaint,
                                   steps, sampler_index, mask_blur, mask_alpha,
                                   inpainting_fill,
                                   restore_faces,
                                   tiling,
                                   n_iter,
                                   _batch_size,  # batch_size
                                   cfg_scale, image_cfg_scale,
                                   denoising_strength, seed,
                                   subseed,
                                   subseed_strength, seed_resize_from_h,
                                   seed_resize_from_w,
                                   seed_enable_extras,
                                   selected_scale_tab, _input_image_height,
                                   _input_image_width,
                                   scale_by,
                                   resize_mode,
                                   inpaint_full_res,
                                   inpaint_full_res_padding, inpainting_mask_invert,
                                   img2img_batch_input_dir,
                                   img2img_batch_output_dir,
                                   img2img_batch_inpaint_mask_dir,
                                   override_settings_texts,
                                   *sam_args)
        self.devices.torch_gc()

        # if _task_type == 'haircut':
        #     if res[0][0].size[0] != _input_image_width:
        #         new_img = Image.new("RGBA", (_input_image_width, res[0][0].size[1]), (0, 0, 0, 1))
        #         new_img.paste(res[0][0], (0,0))
        #         res[0][0] = new_img
        #     if res[0][0].size[1] != _input_image_height:
        #         new_img = Image.new("RGBA", (res[0][0].size[0], _input_image_height), (0, 0, 0, 1))
        #         new_img.paste(res[0][0], (0,0))
        #         res[0][0] = new_img
        #
        #     #
        #     origin_rgba = _init_img.convert('RGBA')
        #     # output face and hair
        #     output_head_result = self.facer.detect_head(res[0][0], return_rect=False)
        #
        #     # origin without hair
        #     sam_hair_result, _ = self.sam.sam_predict(self.dino_model_name, 'hair',
        #                                                     0.3,
        #                                                     origin_rgba)
        #     sam_hair_data = sam_hair_result[1].convert('L').getdata()
        #
        #     # 创建一个新的图像，将原图中mask为0的像素设置为透明
        #     new_image = Image.new("RGBA", origin_rgba.size, (0, 0, 0, 0))
        #
        #     # 遍历每个像素，根据mask的值来设置新图的像素
        #     for y in range(origin_rgba.size[1]):
        #         for x in range(origin_rgba.size[0]):
        #             pixel_value = sam_hair_data[y * origin_rgba.size[0] + x]
        #             if pixel_value == 0:
        #                 new_image.putpixel((x, y), origin_rgba.getpixel((x, y)))
        #
        #     origin_nohair_image = new_image
        #     nohair_array = np.array(origin_nohair_image)
        #     origin_nohair_image.save('nohair.png')
        #
        #     # paste
        #     _output_hair_array = np.array(output_head_result)
        #     _output_hair_rgba = res[0][0].convert('RGBA')
        #     _output_hair_rgba_arr = np.array(_output_hair_rgba)
        #     # 将掩码区域的像素置零
        #     _output_hair_rgba_arr[:, :, 3][_output_hair_array == 0] = 0
        #     # 将结果数组转换为 PIL 图像
        #     _output_hair_crop_image = Image.fromarray(_output_hair_rgba_arr)
        #
        #     origin_nohair_image.paste(_output_hair_crop_image, (0, 0), mask=output_head_result)
        #
        #     origin_nohair_image.save('testing.png')
        #
        #     # mask
        #     # sam_person_result, _ = self.sam.sam_predict(self.dino_model_name, 'person',
        #     #                                               0.3,
        #     #                                               origin_rgba)
        #     # 计算没有相交的 "person" 掩码
        #     # person_array = np.array(sam_person_result[1].convert('L'))
        #     # hair_array = np.array(sam_hair_result[1].convert('L'))
        #     # result_array = np.maximum(person_array - hair_array, 0)
        #     # output_head_array = np.array(output_head_result.convert('L'))
        #     # mask_res = Image.fromarray(np.maximum(result_array, output_head_array))
        #     # 将RGBA图片转换为灰度图（带有透明度通道）
        #     mask_image = origin_nohair_image.convert("LA")
        #
        #     # 提取透明度通道数据
        #     alpha_data = mask_image.split()[1]
        #
        #     # 创建一个新的图像，将透明部分设置为黑色
        #     mask_res = Image.new("L", origin_nohair_image.size, 0)
        #     mask_res.paste(alpha_data, (0, 0), alpha_data)
        #
        #     # mask_res = Image.fromarray(np.maximum(np.array(origin_nohair_image.convert('1')), np.array(output_head_result.convert('1'))))
        #     cache_fp = f"tmp/hair1_{_task_type}_{1}_{uid_name}_{_pic_name}_save.png"
        #     mask_res.save(cache_fp)
        #     sam_result_tmp_png_fp[1] = {'name': cache_fp}
        #     # sam_result_tmp_png_fp[0] = {'name': 'testing.png'}
        #     # sam_result_tmp_png_fp[2] = {'name': 'testing.png'}
        #
        #     _init_img = origin_nohair_image
        #
        #     sam_args = [0,
        #                 adetail_enabled, face_args, hand_args,  # adetail args
        #                 controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,
        #                 # controlnet args
        #                 True, False, 0, _init_img,
        #                 sam_result_tmp_png_fp,
        #                 0,  # sam_output_chosen_mask
        #                 False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
        #                 '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
        #                 True, True, '',
        #                 # tiled diffsuion
        #                 False, 'MultiDiffusion', False, True,
        #                 1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
        #                 64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
        #                 False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
        #                 '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
        #                 False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
        #                 '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
        #                 False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
        #                 # tiled_vae
        #                 False, 256, 48, True, True, True,
        #                 False
        #                 ]
        #     res = self.img2img.img2img(task_id,
        #                                4,
        #                                "(best quality:1.2),(high quality:1.2),(Realism:1.4),masterpiece,raw photo,realistic,character close-up,<lora:more_details:1>",
        #                                "(hair:2.0),"+sd_negative_prompt,
        #                                prompt_styles, _init_img,
        #                                sketch,
        #                                init_img_with_mask, inpaint_color_sketch,
        #                                inpaint_color_sketch_orig,
        #                                init_img_inpaint, init_mask_inpaint,
        #                                steps, sampler_index, mask_blur, mask_alpha,
        #                                inpainting_fill,
        #                                restore_faces,
        #                                tiling,
        #                                n_iter,
        #                                _batch_size,  # batch_size
        #                                cfg_scale, image_cfg_scale,
        #                                # denoising_strength
        #                                0.9,
        #                                seed,
        #                                subseed,
        #                                subseed_strength, seed_resize_from_h,
        #                                seed_resize_from_w,
        #                                seed_enable_extras,
        #                                selected_scale_tab, _input_image_height,
        #                                _input_image_width,
        #                                scale_by,
        #                                resize_mode,
        #                                inpaint_full_res,
        #                                inpaint_full_res_padding, inpainting_mask_invert,
        #                                img2img_batch_input_dir,
        #                                img2img_batch_output_dir,
        #                                img2img_batch_inpaint_mask_dir,
        #                                override_settings_texts,
        #                                *sam_args)
        #
        #     self.devices.torch_gc()

        if return_list:
            return [x.convert('RGBA') for x in res[0]]
        else:
            return res[0][0].convert('RGBA')

    def proceed_avatar(self, _init_img, _selected_index, _selected_type, _gender, _denoising, _batch_size,
                       _txt2img=False):
        uid_name = ''.join([random.choice(string.ascii_letters) for c in range(4)])
        task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

        # prompt setting
        sd_negative_prompt = f"(NSFW:1.8),EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2),bad anatomy, DeepNegative,text, error, cropped, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions"

        prompt_dict = female_avatar_reference_dict if _gender == 'female' else male_avatar_reference_dict
        _selected_style = prompt_dict[_selected_index]['label']
        reference_enbale = True if (_gender == 'female' and _selected_style != "素描") else False

        if reference_enbale:
            _reference_img_rgb_ndarray = np.array(Image.open(
                os.path.join(reference_dir, f"avatar_reference", _gender, _selected_style,
                             f"{str(_selected_type)}.jpeg")).convert('RGB'))
            _reference_img_mask_ndarray = np.zeros(shape=_reference_img_rgb_ndarray.shape)
            sd_positive_prompt = f"{prompt_dict[_selected_index]['prompt'] + ',' if prompt_dict[_selected_index]['prompt'] else ''}<lora:more_details:1>,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,strong contrast,huge_filesize,incredibly_absurdres,absurdres,highres,magazine cover,intense angle,dynamic angle,high saturation,poster"

        else:
            if (_gender == 'female' and _selected_style == '素描') or (_gender == 'male' and (
                    _selected_style == '泥塑' or (_selected_style == '赛博朋克' and _selected_type == 0))):
                sd_positive_prompt = f"{lora_avatar_dict[_selected_style][_selected_type] + ','}{'1boy,' if _gender == 'male' else ''}<lora:more_details:1>,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,strong contrast,huge_filesize,incredibly_absurdres,absurdres,highres,magazine cover,intense angle,dynamic angle,high saturation,poster"
            else:
                if _gender == 'male' and _selected_style == '彩墨':
                    sd_positive_prompt = prompt_dict[_selected_index]['prompt']
                else:
                    sd_positive_prompt = f"{prompt_dict[_selected_index]['prompt'] + ',' if prompt_dict[_selected_index]['prompt'] else ''}{'1boy,' if _gender == 'male' and _selected_style != 'Q版' else ''}<lora:more_details:1>,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,strong contrast,huge_filesize,incredibly_absurdres,absurdres,highres,magazine cover,intense angle,dynamic angle,high saturation,poster"

        # common
        prompt_styles = None
        sketch = None
        init_img_with_mask = None
        inpaint_color_sketch = None
        inpaint_color_sketch_orig = None
        init_img_inpaint = None
        init_mask_inpaint = None
        mask_alpha = 0
        restore_faces = False
        tiling = False
        n_iter = 1
        image_cfg_scale = 1.5
        seed = -1.0
        subseed = -1.0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        selected_scale_tab = 0
        scale_by = 1
        inpaint_full_res_padding = 32
        img2img_batch_input_dir = ''
        img2img_batch_output_dir = ''
        img2img_batch_inpaint_mask_dir = ''
        override_settings_texts = []
        cfg_scale = 5
        # mask_blur = 20
        resize_mode = 0  # just resize
        sampler_index = 15
        inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
        inpainting_fill = 1  # masked content original
        denoising_strength = _denoising
        steps = 20
        mask_blur = 0
        adetail_enabled = False
        face_args = {}
        hand_args = {}

        self.update_progress(50)
        if _txt2img:
            # controlnet args
            controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[self.cnet_idx].get_default_ui_unit()
            # depth
            controlnet_args_unit1.enabled = reference_enbale
            if reference_enbale:
                controlnet_args_unit1.batch_images = ''
                controlnet_args_unit1.guidance_end = 1
                controlnet_args_unit1.guidance_start = 0  # ending control step
                controlnet_args_unit1.low_vram = False
                controlnet_args_unit1.pixel_perfect = False
                controlnet_args_unit1.weight = 1
                controlnet_args_unit1.resize_mode = 'Crop and Resize'
                controlnet_args_unit1.processor_res = 512

                # depth
                controlnet_args_unit1.control_mode = 'Balanced'
                controlnet_args_unit1.image = {
                    'image': _reference_img_rgb_ndarray,
                    'mask': _reference_img_mask_ndarray,
                }
                controlnet_args_unit1.model = 'None'
                # controlnet_args_unit1.module = 'reference_only'
                controlnet_args_unit1.module = 'reference_adain+attn'
                controlnet_args_unit1.threshold_a = 0.5
                controlnet_args_unit1.threshold_b = 64
            controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
            controlnet_args_unit2.enabled = False
            controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
            controlnet_args_unit3.enabled = False
            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                        # sam
                        False,  # inpaint_upload_enable
                        False, 0, _init_img,
                        [],
                        0,  # sam_output_chosen_mask
                        False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

            sd_positive_prompt = f"{sd_positive_prompt},(portrait:1.1),1{'girl' if _gender == 'female' else 'boy'},(half-length:1.1)"
            print("-------------------avatar logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"denoising_strength: {str(1)}")
            print(f"Sampling method: {samplers_k_diffusion[15]}")
            res = self.txt2img.txt2img(task_id,
                                       sd_positive_prompt,
                                       sd_negative_prompt,
                                       prompt_styles,
                                       steps,
                                       sampler_index,
                                       restore_faces,
                                       tiling,
                                       n_iter,
                                       _batch_size,  # batch size
                                       cfg_scale,
                                       seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                       seed_enable_extras,
                                       512,
                                       512,
                                       False,  # enable_hr
                                       0.5,  # denoising_strength
                                       2.0,  # hr_scale
                                       'Latent',  # hr_upscaler"
                                       0,  # hr_second_pass_steps
                                       0,  # hr_resize_x
                                       0,  # hr_resize_y
                                       sampler_index,  # hr_sampler_index
                                       '',  # hr_prompt
                                       '',  # hr_negative_prompt,
                                       override_settings_texts,
                                       *sam_args)

        else:
            _init_img_width, _init_img_height = _init_img.size

            # controlnet args
            _init_img_rgb_ndarray = np.array(_init_img.convert('RGB'))
            _mask_img_ndarray = np.zeros(shape=_init_img_rgb_ndarray.shape)

            controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[self.cnet_idx].get_default_ui_unit()
            controlnet_args_unit1.batch_images = ''
            controlnet_args_unit1.guidance_end = 1
            controlnet_args_unit1.guidance_start = 0  # ending control step
            controlnet_args_unit1.low_vram = False
            controlnet_args_unit1.pixel_perfect = True
            controlnet_args_unit1.weight = 1
            controlnet_args_unit1.enabled = True
            controlnet_args_unit1.resize_mode = 'Crop and Resize'
            controlnet_args_unit1.processor_res = 512
            controlnet_args_unit1.image = {
                'image': _init_img_rgb_ndarray,
                'mask': _mask_img_ndarray,
            }

            # canny
            # controlnet_args_unit1.control_mode = 'ControlNet is more important'
            # controlnet_args_unit1.model = 'control_v11p_sd15_canny'
            # controlnet_args_unit1.module = 'canny'
            # controlnet_args_unit1.threshold_a = 100
            # controlnet_args_unit1.threshold_b = 200

            # depth
            controlnet_args_unit1.control_mode = 'My prompt is more important'
            controlnet_args_unit1.model = 'control_v11f1p_sd15_depth'
            controlnet_args_unit1.module = 'depth_midas'
            controlnet_args_unit1.threshold_a = -1
            controlnet_args_unit1.threshold_b = -1
            controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
            controlnet_args_unit2.enabled = reference_enbale
            if reference_enbale:
                controlnet_args_unit2.image = {
                    'image': _reference_img_rgb_ndarray,
                    'mask': _reference_img_mask_ndarray,
                }
                controlnet_args_unit2.model = 'None'
                controlnet_args_unit2.module = 'reference_only'
                controlnet_args_unit2.processor_res = -1
                controlnet_args_unit2.threshold_a = 1
                controlnet_args_unit2.threshold_b = -1

            print("-------------------avatar logger-----------------")
            print(f"sd_positive_prompt: {sd_positive_prompt}")
            print(f"sd_negative_prompt: {sd_negative_prompt}")
            print(f"denoising_strength: {_denoising}")
            print(f"Sampling method: {samplers_k_diffusion[15]}")
            controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
            controlnet_args_unit3.enabled = False

            # img2img
            inpainting_mask_invert = 0  # 0: inpaint masked 1: inpaint not masked

            sam_args = [0,
                        adetail_enabled, face_args, hand_args,  # adetail args
                        controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,
                        # controlnet args
                        False,  # inpaint_upload_enable
                        False, 0, _init_img,
                        [],
                        0,  # sam_output_chosen_mask
                        False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '',
                        # tiled diffsuion
                        False, 'MultiDiffusion', False, True,
                        1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                        64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                        '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                        # tiled_vae
                        False, 256, 48, True, True, True,
                        False
                        ]

            _input_image_width, _input_image_height = _init_img.size

            res = self.img2img.img2img(task_id,
                                       0,
                                       sd_positive_prompt,
                                       sd_negative_prompt,
                                       prompt_styles, _init_img.convert('RGBA'),
                                       sketch,
                                       init_img_with_mask, inpaint_color_sketch,
                                       inpaint_color_sketch_orig,
                                       init_img_inpaint, init_mask_inpaint,
                                       steps, sampler_index, mask_blur, mask_alpha,
                                       inpainting_fill,
                                       restore_faces,
                                       tiling,
                                       n_iter,
                                       _batch_size,  # batch_size
                                       cfg_scale, image_cfg_scale,
                                       denoising_strength, seed,
                                       subseed,
                                       subseed_strength, seed_resize_from_h,
                                       seed_resize_from_w,
                                       seed_enable_extras,
                                       selected_scale_tab, _input_image_height,
                                       _input_image_width,
                                       scale_by,
                                       resize_mode,
                                       inpaint_full_res,
                                       inpaint_full_res_padding, inpainting_mask_invert,
                                       img2img_batch_input_dir,
                                       img2img_batch_output_dir,
                                       img2img_batch_inpaint_mask_dir,
                                       override_settings_texts,
                                       *sam_args)

        self.devices.torch_gc()

        return [x.convert('L') if _selected_style == '素描' else x.convert('RGBA') for x in res[0]]

    def __call__(self, *args, **kwargs):
        try:
            super().__call__(*args, **kwargs)

            print("operation start !!!!!!!!!!!!!!!!!!!!!!!!!!")
            clean_args = {k: v for k, v in kwargs.items() if k != 'input_image'}
            clean_args['params'] = ujson.loads(kwargs['params'][0])
            print(clean_args)
            proceed_mode = kwargs['mode'][0]
            user_id = kwargs['user_id'][0]
            params = ujson.loads(kwargs['params'][0])

            if self.update_progress(1):
                return {'success': True}
            if self.predict_image(kwargs['input_image']):
                # return {'success': False, 'result': "抱歉，您上传的图像未通过合规性检查，请重新上传。"}
                return {'success': False, 'result': 'backend.check.error.nsfw'}

            # read image
            if proceed_mode != 'wallpaper':
                if 'preset_index' in params.keys() and params['preset_index'] and params['preset_index'] >= 0:
                    _input_image = Image.open(f"guiju/assets/preset/{proceed_mode}/{params['preset_index']}.jpg")
                    _input_image_width, _input_image_height = _input_image.size

                else:
                    _input_image = Image.open(kwargs['input_image'])
                    _input_image_width, _input_image_height = _input_image.size
            else:
                _input_image_width, _input_image_height = 0, 0

            if self.update_progress(5):
                return {'success': True}

            pic_name = ''.join([random.choice(string.ascii_letters) for c in range(6)])

            # logging
            self.logging(
                f"[__call__][{datetime.datetime.now()}]:\n"
                f"[{pic_name}]:\n"
                f"{ujson.dumps(clean_args, indent=4)}",
                f"logs/sd_webui.log")

            if proceed_mode == 'cert':
                _bg_color = str(params['bg_color'])
                _output_aspect = float(params['aspect'])

                # save cache face img
                _input_image.save(f"tmp/cert_origin_{pic_name}_save.png")
                _input_image = _input_image.convert('RGBA')

                if self.update_progress(10):
                    return {'success': True}
                # parse face
                face_boxes = self.facer.detect_face(_input_image)
                if len(face_boxes) == 0:
                    # return {'success': False, 'result': '未检测到人脸'}
                    return {'success': False, 'result': 'backend.magic-avatar.error.no-face'}

                elif len(face_boxes) > 1:
                    # return {'success': False, 'result': '检测到多个人脸，请上传一张单人照'}
                    return {'success': False, 'result': 'backend.magic-avatar.error.multi-face'}

                else:
                    if self.update_progress(30):
                        return {'success': True}

                    # segment person
                    sam_person_result, person_boxes = self.sam.sam_predict(self.dino_model_name, 'person', 0.3,
                                                                           _input_image)
                    padding = 4
                    resize_mask = sam_person_result[1].resize((_input_image_width-padding, _input_image_height-padding))
                    padding_mask = Image.new("RGB", _input_image.size, (0, 0, 0, 1))
                    padding_mask.paste(resize_mask, (int(padding/2), int(padding/2)))
                    padding_mask = padding_mask.convert('1')

                    padding_sam = Image.new('RGBA', _input_image.size)
                    padding_sam.paste(sam_person_result[2], (0, 0), padding_mask)
                    sam_person_result[2] = padding_sam

                if self.update_progress(60):
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
                img_urls = []
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_cert_dir'], user_id)
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
                self.devices.torch_gc()
                pp = self.scripts_postprocessing.PostprocessedImage(cert_res.convert('RGB'))
                self.scripts.scripts_postproc.run(pp, args)
                pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
                self.devices.torch_gc()

                # 限制缓存10张
                cache_list = sorted(os.listdir(dir_path))
                if len(cache_list) > 10:
                    os.remove(os.path.join(dir_path, cache_list[0]))

                for img_fn in sorted(os.listdir(dir_path), reverse=True):
                    url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=cert"
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

                return {'success': True, 'result': img_urls}

            elif proceed_mode == 'mirage':
                if self.shared.sd_model.sd_checkpoint_info.model_name != 'dreamshaper_8':
                    # self.shared.change_sd_model('dreamshaper_8Inpainting')
                    self.shared.change_sd_model('dreamshaper_8')

                origin_image_path = f'tmp/mirage_origin_{pic_name}_save.png'
                _input_image.save(origin_image_path, format='PNG')

                _batch_size = int(params['batch_size'])
                _selected_place = int(params['place'])

                # limit 512
                min_edge = min(_input_image_width, _input_image_height)
                min_index = [_input_image_width, _input_image_height].index(min_edge)
                if min_index == 0:
                    _input_image = _input_image.resize((512, int(_input_image_height / _input_image_width * 512)))
                else:
                    _input_image = _input_image.resize((int(_input_image_width / _input_image_height * 512), 512))

                cache_fp = f"tmp/mirage_resized_{pic_name}_save.png"
                _input_image.save(cache_fp)

                _input_image = _input_image.convert('RGBA')
                _input_image_width, _input_image_height = _input_image.size

                # sam predict person
                sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, 'person.bag',
                                                                0.6, _input_image)

                if len(sam_result) > 0:
                    sam_image = sam_result[2]
                    mask_image = sam_result[1]
                    sam_result[0].save(
                        f"tmp/mirage_sam_{pic_name}_save.png",
                        format='PNG')

                    sam_result_tmp_png_fp = []
                    for resized_img_type, cache_image in zip(["resized_input", "resized_mask", "resized_clothing"],
                                                             [sam_result[0], mask_image, sam_image]):
                        cache_fp = f"tmp/mirage_{resized_img_type}_{pic_name}.png"
                        cache_image.save(cache_fp)
                        sam_result_tmp_png_fp.append({'name': cache_fp})
                else:
                    return {'success': False, 'result': 'backend.magic-mirage.error.no-person'}

                if self.update_progress(20):
                    return {'success': True}

                # img2img generate bg
                prompt_styles = None
                # init_img = sam_image
                init_img = _input_image

                sketch = None
                init_img_with_mask = None
                inpaint_color_sketch = None
                inpaint_color_sketch_orig = None
                init_img_inpaint = None
                init_mask_inpaint = None
                steps = 20
                sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
                mask_blur = 0
                mask_alpha = 0
                inpainting_fill = 1
                restore_faces = False
                tiling = False
                n_iter = 1
                batch_size = _batch_size
                cfg_scale = 10
                image_cfg_scale = 1.5
                denoising_strength = 0.75
                seed = -1.0
                subseed = -1.0
                subseed_strength = 0
                seed_resize_from_h = 0
                seed_resize_from_w = 0
                seed_enable_extras = False
                selected_scale_tab = 0
                scale_by = 1
                resize_mode = 0  # 1: crop and resize 2: resize and fill
                inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
                inpaint_full_res_padding = 32
                inpainting_mask_invert = 1  # Mask mode 0: Inpaint masked - 1: Inpaint not masked
                img2img_batch_input_dir = ''
                img2img_batch_output_dir = ''
                img2img_batch_inpaint_mask_dir = ''
                override_settings_texts = []

                # controlnet args
                controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[
                    self.cnet_idx].get_default_ui_unit()
                controlnet_args_unit1.batch_images = ''
                controlnet_args_unit1.control_mode = 'My prompt is more important'
                controlnet_args_unit1.enabled = True
                controlnet_args_unit1.guidance_end = 1
                controlnet_args_unit1.guidance_start = 0  # ending control step
                controlnet_args_unit1.low_vram = False
                controlnet_args_unit1.loopback = False
                controlnet_args_unit1.processor_res = 512
                controlnet_args_unit1.threshold_a = 0.5
                controlnet_args_unit1.threshold_b = -1
                controlnet_args_unit1.model = 'None'
                controlnet_args_unit1.module = 'reference_adain+attn'
                controlnet_args_unit1.pixel_perfect = True
                controlnet_args_unit1.weight = 1
                controlnet_args_unit1.resize_mode = 'Crop and Resize'

                _reference_dir_path = os.path.join(reference_dir, "mirage_reference", str(_selected_place))
                _reference_image_path = os.path.join(_reference_dir_path,
                                                     f'{random.randint(0, len(os.listdir(_reference_dir_path)) - 1)}.jpeg')
                self.logging(
                    f"[_reference_image_path][{_reference_image_path}]:\n",
                    f"logs/sd_webui.log")
                _reference_img_rgb_ndarray = np.array(Image.open(_reference_image_path))
                _reference_img_mask_ndarray = np.zeros(shape=_reference_img_rgb_ndarray.shape)
                controlnet_args_unit1.image = {
                    'image': _reference_img_rgb_ndarray,
                    'mask': _reference_img_mask_ndarray,
                }

                controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit2.enabled = True
                controlnet_args_unit2.model = 'control_v11e_sd15_shuffle'
                controlnet_args_unit2.module = 'shuffle'
                controlnet_args_unit2.control_mode = 'Balanced'
                controlnet_args_unit2.threshold_a = -1
                controlnet_args_unit2.threshold_b = -1

                # depth
                controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit3.enabled = True
                controlnet_args_unit3.processor_res = 512
                controlnet_args_unit3.image = {
                    'image': np.array(_input_image),
                    'mask': np.zeros(shape=(_input_image_height, _input_image_width)),
                }
                controlnet_args_unit3.control_mode = 'Balanced'
                controlnet_args_unit3.model = 'control_v11f1p_sd15_depth'
                controlnet_args_unit3.module = 'depth_midas'
                controlnet_args_unit3.threshold_a = -1
                controlnet_args_unit3.threshold_b = -1

                # adetail
                adetail_enabled = False
                face_args = {}
                hand_args = {}
                sam_args = [0,
                            adetail_enabled, face_args, hand_args,  # adetail args
                            controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                            # sam
                            True, False, 0, _input_image,
                            sam_result_tmp_png_fp,
                            0,  # sam_output_chosen_mask
                            False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                            False, 0, None, None,
                            # tiled diffsuion
                            False if _selected_place == 12 or _selected_place == 6 else True, 'MultiDiffusion', False,
                            True, 1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                            64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                            '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                            '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            # tiled_vae
                            False if _selected_place == 12 or _selected_place == 6 else True, 256, 48, True, True, True,
                            False
                            ]

                # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
                if self.update_progress(50):
                    return {'success': True}

                _output_model_width, _output_model_height = init_img.size
                task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
                sd_positive_prompt = f"{lora_mirage_dict[_selected_place]['prompt']},<lora:more_details:1>,science fiction,fantasy,incredible,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,(dramatic scene),(Epic composition:1.2),strong contrast,no humans,magazine cover,intense angle,dynamic angle,high saturation,poster"
                sd_negative_prompt = "(NSFW:1.8),(hands),(human),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(hair:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, facing away, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions"

                print("-------------------mirage logger-----------------")
                print(f"sd_positive_prompt: {sd_positive_prompt}")
                print(f"sd_negative_prompt: {sd_negative_prompt}")
                print(f"dino_prompt: person")
                print(f"denoising_strength: {denoising_strength}")
                print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")
                # 生成
                res = self.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt,
                                           prompt_styles,
                                           init_img,
                                           sketch,
                                           init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                           init_img_inpaint, init_mask_inpaint,
                                           steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                           restore_faces,
                                           tiling,
                                           n_iter, batch_size, cfg_scale, image_cfg_scale,
                                           denoising_strength,
                                           seed,
                                           subseed,
                                           subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                           seed_enable_extras,
                                           selected_scale_tab, _output_model_height, _output_model_width, scale_by,
                                           resize_mode,
                                           inpaint_full_res,
                                           inpaint_full_res_padding, inpainting_mask_invert,
                                           img2img_batch_input_dir,
                                           img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                           override_settings_texts,
                                           *sam_args)[0]

                # storage img
                img_urls = []
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_mirage_dir'], user_id)
                os.makedirs(dir_path, exist_ok=True)
                for res_idx, res_img in enumerate(res):
                    img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                    res_img.convert("RGB").save(os.path.join(dir_path, img_fn), format="jpeg", quality=80,
                                                lossless=True)

                    # 限制缓存10张
                    cache_list = sorted(os.listdir(dir_path))
                    if len(cache_list) > 10:
                        os.remove(os.path.join(dir_path, cache_list[0]))
                else:
                    for img_fn in sorted(os.listdir(dir_path), reverse=True):
                        url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=mirage"
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

                return {'success': True, 'result': img_urls}

            elif proceed_mode == 'wallpaper':
                if self.shared.sd_model.sd_checkpoint_info.model_name != 'dreamshaper_8':
                    self.shared.change_sd_model('dreamshaper_8')

                _batch_size = int(params['batch_size'])
                _selected_place = int(params['place'])
                _output_width = int(params['width'])
                _output_height = int(params['height'])
                _selected_aspect = _output_width / _output_height

                # limit 512
                min_edge = 512
                _buf_width = 512
                _buf_height = 512
                if _output_width <= _output_height:
                    _buf_width = min_edge
                    _buf_height = int(min_edge / _selected_aspect)
                else:
                    _buf_width = int(min_edge * _selected_aspect)
                    _buf_height = min_edge

                # target_short_side = 512
                # closest_divisor = 1
                # closest_remainder = float('inf')
                #
                # for i in range(1, min(_output_width, _output_height) + 1):
                #     if _output_width % i == 0 and _output_height % i == 0:
                #         short_side = min(_output_width // i, _output_height // i)
                #         remainder = abs(target_short_side - short_side)
                #         if remainder < closest_remainder:
                #             closest_remainder = remainder
                #             closest_divisor = i
                #
                # _buf_width = _output_width//closest_divisor
                # _buf_height = _output_height//closest_divisor

                print(f"_buf_width:{_buf_width}")
                print(f"_buf_height:{_buf_height}")

                if self.update_progress(20):
                    return {'success': True}

                # img2img generate bg
                prompt_styles = None
                steps = 20
                sampler_index = 15  # sampling method modules/sd_samplers_kdiffusion.py
                restore_faces = False
                tiling = False
                n_iter = 1
                cfg_scale = 10
                seed = -1.0
                subseed = -1.0
                subseed_strength = 0
                seed_resize_from_h = 0
                seed_resize_from_w = 0
                seed_enable_extras = False
                override_settings_texts = []

                # controlnet args
                controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[
                    self.cnet_idx].get_default_ui_unit()
                controlnet_args_unit1.batch_images = ''
                controlnet_args_unit1.control_mode = 'My prompt is more important'
                controlnet_args_unit1.enabled = True
                controlnet_args_unit1.guidance_end = 1
                controlnet_args_unit1.guidance_start = 0  # ending control step
                controlnet_args_unit1.low_vram = False
                controlnet_args_unit1.loopback = False
                controlnet_args_unit1.processor_res = 512
                controlnet_args_unit1.threshold_a = 0.5
                controlnet_args_unit1.threshold_b = -1
                controlnet_args_unit1.model = 'None'
                controlnet_args_unit1.module = 'reference_adain+attn'
                controlnet_args_unit1.pixel_perfect = True
                controlnet_args_unit1.weight = 1
                controlnet_args_unit1.resize_mode = 'Crop and Resize'

                _reference_dir_path = os.path.join(reference_dir, "mirage_reference", str(_selected_place))
                _reference_image_path = os.path.join(_reference_dir_path,
                                                     f'{random.randint(0, len(os.listdir(_reference_dir_path)) - 1)}.jpeg')
                self.logging(
                    f"[_reference_image_path][{_reference_image_path}]:\n",
                    f"logs/sd_webui.log")
                _reference_img_rgb_ndarray = np.array(Image.open(_reference_image_path))
                _reference_img_mask_ndarray = np.zeros(shape=_reference_img_rgb_ndarray.shape)
                controlnet_args_unit1.image = {
                    'image': _reference_img_rgb_ndarray,
                    'mask': _reference_img_mask_ndarray,
                }

                controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit2.enabled = True
                controlnet_args_unit2.model = 'control_v11e_sd15_shuffle'
                controlnet_args_unit2.module = 'shuffle'
                controlnet_args_unit2.control_mode = 'Balanced'
                controlnet_args_unit2.threshold_a = -1
                controlnet_args_unit2.threshold_b = -1

                # depth
                controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit3.enabled = False

                # adetail
                adetail_enabled = False
                face_args = {}
                hand_args = {}
                sam_args = [0,
                            adetail_enabled, face_args, hand_args,  # adetail args
                            controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                            # sam
                            False, False, 0, None,
                            [],
                            0,  # sam_output_chosen_mask
                            False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                            False, 0, None, None,
                            # tiled diffsuion
                            False if _selected_place == 12 or _selected_place == 6 else True, 'MultiDiffusion', False,
                            True, 1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                            64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                            '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                            '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            # tiled_vae
                            False if _selected_place == 12 or _selected_place == 6 else True, 256, 48, True, True, True,
                            False
                            ]

                # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
                if self.update_progress(50):
                    return {'success': True}

                task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
                sd_positive_prompt = f"{lora_mirage_dict[_selected_place]['prompt']},<lora:more_details:1>,science fiction,fantasy,incredible,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,(dramatic scene),(Epic composition:1.2),strong contrast,no humans,magazine cover,intense angle,dynamic angle,high saturation,poster"
                sd_negative_prompt = "(NSFW:1.8),(hands),(human),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(hair:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, facing away, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions"

                print("-------------------wallpaper logger-----------------")
                print(f"sd_positive_prompt: {sd_positive_prompt}")
                print(f"sd_negative_prompt: {sd_negative_prompt}")
                print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")
                # 生成
                res = self.txt2img.txt2img(task_id,
                                           sd_positive_prompt,
                                           sd_negative_prompt,
                                           prompt_styles,
                                           steps,
                                           sampler_index,
                                           restore_faces,
                                           tiling,
                                           n_iter,
                                           _batch_size,  # batch size
                                           cfg_scale,
                                           seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                           seed_enable_extras,
                                           _buf_height,
                                           _buf_width,
                                           False,  # enable_hr
                                           0.5,  # denoising_strength
                                           2.0,  # hr_scale
                                           'Latent',  # hr_upscaler"
                                           0,  # hr_second_pass_steps
                                           0,  # hr_resize_x
                                           0,  # hr_resize_y
                                           sampler_index,  # hr_sampler_index
                                           '',  # hr_prompt
                                           '',  # hr_negative_prompt,
                                           override_settings_texts,
                                           *sam_args)[0]

                # storage img
                img_urls = []
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_wallpaper_dir'], user_id)
                os.makedirs(dir_path, exist_ok=True)
                for res_idx, res_img in enumerate(res):
                    # hires
                    # extra upscaler
                    scales = _output_width / _buf_width
                    gfpgan_weight = 0
                    codeformer_visibility = 0

                    upscale_mode = 1  # 0:scale_by 1:scale_to
                    upscale_by = None
                    upscale_to_width = _output_width
                    upscale_to_height = _output_height
                    upscale_crop = True
                    upscaler_1_name = 'R-ESRGAN 4x+'
                    upscaler_2_name = 'None'
                    upscaler_2_visibility = 0
                    args = (
                        upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop, upscaler_1_name,
                        upscaler_2_name, upscaler_2_visibility,
                        gfpgan_weight, codeformer_visibility,
                        0)
                    self.devices.torch_gc()
                    pp = self.scripts_postprocessing.PostprocessedImage(res_img.convert("RGB"))
                    self.scripts.scripts_postproc.run(pp, args)

                    self.devices.torch_gc()

                    img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                    # res_img.convert("RGB").save(os.path.join(dir_path, img_fn), format="jpeg", quality=80,
                    #                             lossless=True)
                    pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=100, lossless=True)

                    # 限制缓存10张
                    cache_list = sorted(os.listdir(dir_path))
                    if len(cache_list) > 10:
                        os.remove(os.path.join(dir_path, cache_list[0]))
                else:
                    for img_fn in sorted(os.listdir(dir_path), reverse=True):
                        url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=wallpaper"
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

                return {'success': True, 'result': img_urls}

            elif proceed_mode == 'facer':
                if self.update_progress(40):
                    return {'success': True}
                # _batch_size = int(params['batch_size'])
                _input_src_image = cv2.imread(kwargs['input_image'])
                _input_tgt_image = cv2.imread(kwargs['input_image_tgt'])

                if self.update_progress(60):
                    return {'success': True}
                src_faces = self.face_analysis.get(_input_src_image)
                if len(src_faces) != 1:
                    # return {'success': False, 'result': '未检测到人脸'}
                    return {'success': False, 'result': 'backend.magic-facer.error.no-face0'}
                tgt_faces = self.face_analysis.get(_input_tgt_image)
                if len(tgt_faces) != 1:
                    # return {'success': False, 'result': '未检测到人脸'}
                    return {'success': False, 'result': 'backend.magic-facer.error.no-face1'}

                res = self.swapper.get(_input_tgt_image, tgt_faces[0], src_faces[0], paste_back=True)
                # storage img
                img_urls = []
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_facer_dir'], user_id)
                os.makedirs(dir_path, exist_ok=True)

                img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                cv2.imwrite(os.path.join(dir_path, img_fn), res)

                if self.update_progress(80):
                    return {'success': True}
                # face fix
                gfpgan_weight = 0.5
                scales = 1
                codeformer_weight = 0
                codeformer_visibility = 0
                args = (0, scales, None, None, True, 'ESRGAN_4x', 'None',
                        0, gfpgan_weight,
                        codeformer_visibility, codeformer_weight)
                self.devices.torch_gc()
                pp = self.scripts_postprocessing.PostprocessedImage(Image.open(os.path.join(dir_path, img_fn)))
                self.scripts.scripts_postproc.run(pp, args)
                pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
                self.devices.torch_gc()

                # 限制缓存10张
                cache_list = sorted(os.listdir(dir_path))
                if len(cache_list) > 10:
                    os.remove(os.path.join(dir_path, cache_list[0]))

                for img_fn in sorted(os.listdir(dir_path), reverse=True):
                    url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=facer"
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

                return {'success': True, 'result': img_urls}

            elif proceed_mode == 'avatar':
                # if self.shared.sd_model.sd_checkpoint_info.model_name == 'dreamshaper_8':
                #     self.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')
                if self.shared.sd_model.sd_checkpoint_info.model_name != 'dreamshaper_8':
                    self.shared.change_sd_model('dreamshaper_8')

                _batch_size = int(params['batch_size'])
                _style = int(params['style'])
                _sim = float(params['sim'])
                _type = int(params['type'])
                _gender = str(params['gender'])
                # _txt2img = bool(params['txt2img'])
                _txt2img = False

                # 0.2–0.4
                denoising_strength_min = 0.3
                denoising_strength_max = 0.4
                denoising_strength = (1 - _sim) * (
                        denoising_strength_max - denoising_strength_min) + denoising_strength_min

                if _txt2img:
                    _input_image = None

                else:
                    person_boxes = self.facer.detect_face(_input_image)
                    if len(person_boxes) == 0:
                        # return {'success': False, 'result': '未检测到人脸'}
                        return {'success': False, 'result': 'backend.magic-avatar.error.no-face'}

                    elif len(person_boxes) > 1:
                        # return {'success': False, 'result': '检测到多个人脸，请上传一张单人照'}
                        return {'success': False, 'result': 'backend.magic-avatar.error.multi-face'}

                    # save cache face img
                    cache_image = _input_image.copy()
                    draw = ImageDraw.Draw(cache_image)
                    draw.rectangle(person_boxes[0], outline='red', width=5)
                    cache_image.save(
                        f"tmp/avatar_origin_face_style{str(_style)}_type{_type}_sim{_sim}_gender{_gender}_{pic_name}_save.png")

                    # get max area clothing box
                    person_box = person_boxes[0]
                    person_width = person_box[2] - person_box[0]
                    person_height = person_box[3] - person_box[1]

                    padding_ratio = 1
                    person_box[0] = person_box[0] - int(person_width * padding_ratio)
                    if person_box[0] < 0:
                        person_box[0] = 0
                    person_box[1] = person_box[1] - int(person_height * padding_ratio)
                    if person_box[1] < 0:
                        person_box[1] = 0
                    person_box[2] = person_box[2] + int(person_width * padding_ratio)
                    if person_box[2] >= _input_image_width:
                        person_box[2] = _input_image_width - 1
                    person_box[3] = person_box[3] + int(person_height * padding_ratio)
                    if person_box[3] >= _input_image_height:
                        person_box[3] = _input_image_height - 1

                    # 正方形
                    person_width = person_box[2] - person_box[0]
                    person_height = person_box[3] - person_box[1]
                    if person_width < person_height:
                        padding_left = int((person_height - person_width) / 2)
                        person_box[0] = person_box[0] - padding_left
                        if person_box[0] < 0:
                            person_box[0] = 0
                        padding_right = person_height - person_width - padding_left
                        person_box[2] = person_box[2] + padding_right
                        if person_box[2] > _input_image_width:
                            person_box[2] = _input_image_width

                    elif person_width > person_height:
                        padding_top = int((person_width - person_height) / 2)
                        person_box[1] = person_box[1] - padding_top
                        if person_box[1] < 0:
                            person_box[1] = 0
                        padding_bottom = person_width - person_height - padding_top
                        person_box[3] = person_box[3] + padding_bottom
                        if person_box[3] > _input_image_height:
                            person_box[3] = _input_image_height

                    # crop
                    _input_image = _input_image.crop(person_box)

                    _input_image_width, _input_image_height = _input_image.size
                    # limit 512
                    min_edge = min(_input_image_width, _input_image_height)
                    min_index = [_input_image_width, _input_image_height].index(min_edge)
                    if min_index == 0:
                        _input_image = _input_image.resize((512, int(_input_image_height / _input_image_width * 512)))
                    else:
                        _input_image = _input_image.resize((int(_input_image_width / _input_image_height * 512), 512))

                    cache_fp = f"tmp/avatar_resized_{pic_name}_save.png"
                    _input_image.save(cache_fp)

                    _input_image = _input_image.convert('RGBA')
                    _input_image_width, _input_image_height = _input_image.size

                if self.update_progress(10):
                    return {'success': True}

                avatar_result = self.proceed_avatar(_input_image, _style, _type, _gender, denoising_strength,
                                                    _batch_size, _txt2img)

                # storage img
                img_urls = []
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_avatar_dir'], user_id)
                os.makedirs(dir_path, exist_ok=True)
                for res_idx, res_img in enumerate(avatar_result):
                    img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                    res_img.convert("RGB").save(os.path.join(dir_path, img_fn), format="jpeg", quality=80,
                                                lossless=True)

                    # 限制缓存10张
                    cache_list = sorted(os.listdir(dir_path))
                    if len(cache_list) > 10:
                        os.remove(os.path.join(dir_path, cache_list[0]))
                else:
                    for img_fn in sorted(os.listdir(dir_path), reverse=True):
                        url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=avatar"
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

                return {'success': True, 'result': img_urls}

            elif proceed_mode == 'hair':
                if self.shared.sd_model.sd_checkpoint_info.model_name != 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                    self.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')
                # elif self.shared.sd_model.sd_checkpoint_info.model_name == 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                #     self.shared.change_sd_model('dreamshaper_8')

                _batch_size = int(params['batch_size'])
                _haircut_style = int(params['haircut'])
                _hair_color = int(params['hair_color'])
                _haircut_enable = bool(params['haircut_enable'])
                _hair_color_enable = bool(params['hair_color_enable'])
                _gender = str(params['gender'])

                if _input_image is None:
                    # return {'success': False, 'result': '未接收到图片'}
                    return {'success': False, 'result': 'backend.check.error.no-image'}

                else:
                    origin_image_path = f'tmp/hair_origin_{pic_name}_save.png'
                    _input_image.save(origin_image_path, format='PNG')

                    # preprocess
                    # sam_result, person_boxes = self.sam_h.sam_predict(self.dino_model_name, 'face',
                    #                                                   0.43,
                    #                                                   _input_image.convert('RGBA'))

                    if _hair_color_enable:
                        person_boxes = self.facer.detect_head(_input_image)
                    else:
                        person_boxes = self.facer.detect_face(_input_image)

                    if len(person_boxes) == 0:
                        # return {'success': False, 'result': '未检测到人脸'}
                        return {'success': False, 'result': 'backend.magic-hair.error.no-face'}

                    elif len(person_boxes) > 1:
                        # return {'success': False, 'result': '检测到多个人脸，请上传一张单人照'}
                        return {'success': False, 'result': 'backend.magic-hair.error.multi-face'}

                    else:
                        # save cache face img
                        cache_image = _input_image.copy()
                        draw = ImageDraw.Draw(cache_image)
                        draw.rectangle(person_boxes[0], outline='red', width=5)
                        cache_image.save(f"tmp/hair_face_{pic_name}_save.png")

                        # get max area clothing box
                        # x_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 0 or i == 2]
                        # y_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 1 or i == 3]
                        # person_box = [min(x_list), min(y_list), max(x_list), max(y_list)]
                        # person_width = person_box[2] - person_box[0]
                        # person_height = person_box[3] - person_box[1]
                        person_box = person_boxes[0]
                        person_width = person_box[2] - person_box[0]
                        person_height = person_box[3] - person_box[1]

                        new_person_box = [0, 0, 0, 0]

                        # crop
                        if _hair_color_enable:
                            new_person_box[0] = person_box[0] - int(person_width * 0.6)
                            new_person_box[1] = person_box[1] - int(person_height * 0.6)
                            new_person_box[2] = person_box[2] + int(person_width * 0.6)
                            new_person_box[3] = person_box[3] + int(person_height * 0.6)
                            if new_person_box[0] < 0:
                                new_person_box[0] = 0
                            if new_person_box[1] < 0:
                                new_person_box[1] = 0
                            if new_person_box[2] >= _input_image_width:
                                new_person_box[2] = _input_image_width - 1
                            if new_person_box[3] >= _input_image_height:
                                new_person_box[3] = _input_image_height - 1
                            _input_image = _input_image.crop(new_person_box)
                        else:
                            new_person_box[0] = person_box[0] - int(person_width * 0.6)
                            new_person_box[1] = person_box[1] - int(person_height * 0.6)
                            new_person_box[2] = person_box[2] + int(person_width * 0.6)
                            new_person_box[3] = person_box[3] + int(person_height * 0.8)
                            if new_person_box[0] < 0:
                                new_person_box[0] = 0
                            if new_person_box[1] < 0:
                                new_person_box[1] = 0
                            if new_person_box[2] > _input_image_width - 1:
                                new_person_box[2] = _input_image_width - 1
                            if new_person_box[3] > _input_image_height - 1:
                                new_person_box[3] = _input_image_height - 1

                            # need_padding = True if new_person_box[0] < 0 or new_person_box[1] < 0 or new_person_box[
                            #     2] > _input_image_width - 1 or new_person_box[3] > _input_image_height - 1 else False

                            # if need_padding:
                            #     # _input_image = _input_image.crop(person_box)
                            #     new_image_width = new_person_box[2] - new_person_box[0]
                            #     new_image_height = new_person_box[3] - new_person_box[1]
                            #     new_canvas = Image.new("RGBA", (new_image_width, new_image_height), (0, 0, 0, 0))
                            #
                            #     origin_box_x = abs(new_person_box[0]) if new_person_box[0] < 0 else 0
                            #     origin_box_y = abs(new_person_box[1]) if new_person_box[1] < 0 else 0
                            #
                            #     if new_person_box[0] >= 0 or new_person_box[1] >= 0:
                            #         _input_image = _input_image.crop([0 if new_person_box[0] < 0 else new_person_box[0],
                            #                                           0 if new_person_box[1] < 0 else new_person_box[1],
                            #                                           _input_image_width - 1,
                            #                                           _input_image_height - 1])
                            #
                            #     new_canvas.paste(_input_image, (origin_box_x, origin_box_y))
                            #     _input_image = new_canvas
                            # else:
                            _input_image = _input_image.crop(new_person_box)

                        _input_image_width, _input_image_height = _input_image.size
                        # limit 512
                        min_edge = min(_input_image_width, _input_image_height)
                        min_index = [_input_image_width, _input_image_height].index(min_edge)
                        if min_index == 0:
                            _input_image = _input_image.resize(
                                (512, int(_input_image_height / _input_image_width * 512)))
                        else:
                            _input_image = _input_image.resize(
                                (int(_input_image_width / _input_image_height * 512), 512))

                        cache_fp = f"tmp/hair_resized_{pic_name}_save.png"
                        _input_image.save(cache_fp)

                    _input_image = _input_image.convert('RGBA')
                    _input_image_width, _input_image_height = _input_image.size

                    if self.update_progress(10):
                        return {'success': True}

                    hair_result = []
                    # haircut
                    if _haircut_enable:
                        hair_result = self.proceed_hair(_haircut_style, 'haircut', _batch_size, _input_image, pic_name,
                                                        return_list=True, gender=_gender)
                        if isinstance(hair_result, dict):
                            return hair_result

                    if self.update_progress(50):
                        return {'success': True}

                    # hair color
                    if _hair_color_enable:
                        if _haircut_enable:
                            for index, haircut_res in enumerate(hair_result):
                                cache_fp = f"tmp/hair_haircut_res_{index}_{pic_name}_save.png"
                                haircut_res.save(cache_fp)

                                hair_color_res = self.proceed_hair(_hair_color, 'hair_color', 1, haircut_res, pic_name,
                                                                   return_list=False)
                                if isinstance(hair_color_res, dict):
                                    return hair_color_res
                                else:
                                    hair_result[index] = hair_color_res


                        else:
                            hair_result = self.proceed_hair(_hair_color, 'hair_color', _batch_size, _input_image,
                                                            pic_name, return_list=True)
                            if isinstance(hair_result, dict):
                                return hair_result

                    # storage img
                    img_urls = []
                    dir_path = os.path.join(CONFIG['storage_dirpath']['user_hair_dir'], user_id)
                    os.makedirs(dir_path, exist_ok=True)
                    for res_idx, res_img in enumerate(hair_result):
                        img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                        res_img.convert("RGB").save(os.path.join(dir_path, img_fn), format="jpeg", quality=80,
                                                    lossless=True)

                        # 限制缓存10张
                        cache_list = sorted(os.listdir(dir_path))
                        if len(cache_list) > 10:
                            os.remove(os.path.join(dir_path, cache_list[0]))
                    else:
                        for img_fn in sorted(os.listdir(dir_path), reverse=True):
                            url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=hair"
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

                    return {'success': True, 'result': img_urls}

            # 生成服装模特
            elif proceed_mode == 'model':
                if self.shared.sd_model.sd_checkpoint_info.model_name != 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                    self.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')
                # elif self.shared.sd_model.sd_checkpoint_info.model_name == 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                #     self.shared.change_sd_model('dreamshaper_8')

                _cloth_part = 0
                _batch_size = int(params['batch_size'])
                arge_idxs = {v: i for i, v in enumerate(['child', 'youth', 'middlescent'])}
                _age = arge_idxs[params['age']]
                viewpoint_mode_idxs = {v: i for i, v in enumerate(['front', 'side', 'back'])}
                _viewpoint_mode = viewpoint_mode_idxs[params['viewpoint_mode']]
                _model_mode = 0 if params['model_mode'] == 'normal' else 1  # 0:模特 ,1:人台
                # model and place type
                _model_type = int(params['model_type'])  # 模特类型
                _place_type = int(params['place_type'])  # 背景

                # _output_model_height = 1024
                # _output_model_width = 512
                # _output_final_height = 768
                # _output_final_width = 512

                # _sam_model_name = 'samhq_vit_h_1b3123.pth'

                # _dino_clothing_text_prompt = 'clothing . pants . shorts . dress . shirt . t-shirt . skirt . underwear . bra . swimsuits . bikini . stocking . chain . bow' if _model_mode == 1 else 'clothing . pants . shorts'
                # underwear . bikini和bowtie冲突，bra和bowtie不冲突，考虑分组遍历后在合并
                # _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . bra . bowtie'
                # bikini和t-shirt冲突
                # _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear'
                _dino_clothing_text_prompt = [
                    'clothing . pants . dress . shirt',
                    'bra . bikini . bowtie . chain . underwear . t-shirt',
                ]
                # _dino_clothing_text_prompt_0 = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear'
                # _dino_clothing_text_prompt_1 = 'bra . bikini . bowtie . stocking . chain'
                _box_threshold = 0.35

                if _input_image is None:
                    # return {'success': False, 'result': '未接收到图片'}
                    return {'success': False, 'result': 'backend.check.error.no-image'}

                else:
                    origin_image_path = f'tmp/model_origin_{pic_name}_save.png'
                    _input_image.save(origin_image_path, format='PNG')

                    try:
                        if self.update_progress(10):
                            return {'success': True}

                        # 切割衣服
                        person0_box = [-1, -1, -1, -1]
                        sam_images = []
                        mask_images = []
                        for dino_idx, dino_prompt in enumerate(_dino_clothing_text_prompt):
                            sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, dino_prompt,
                                                                            _box_threshold,
                                                                            _input_image.convert('RGBA'))

                            if len(sam_result) > 0:
                                for idx, im in enumerate(sam_result):
                                    im.save(
                                        f"tmp/model_clothing_{dino_idx}_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png",
                                        format='PNG')
                                sam_images.append(sam_result[2])
                                mask_images.append(sam_result[1])
                                # get max area clothing box
                                x_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 0 or i == 2]
                                y_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 1 or i == 3]
                                box = [min(x_list), min(y_list), max(x_list), max(y_list)]
                                person0_box = [
                                    box[0] if person0_box[0] == -1 or box[0] < person0_box[0] else person0_box[0],
                                    box[1] if person0_box[1] == -1 or box[1] < person0_box[1] else person0_box[1],
                                    box[2] if person0_box[2] == -1 or box[2] > person0_box[2] else person0_box[2],
                                    box[3] if person0_box[3] == -1 or box[3] > person0_box[3] else person0_box[3],
                                ]

                        if self.update_progress(20):
                            return {'success': True}

                        if person0_box[0] == -1:
                            # return {'success': False, 'result': '未检测到服装'}
                            return {'success': False, 'result': 'backend.magic-closet.error.no-cloth'}

                        else:
                            clothing_image = Image.new("RGBA", (_input_image_width, _input_image_height),
                                                       (255, 255, 255, 1))
                            mask_image = Image.new("RGBA", (_input_image_width, _input_image_height), (0, 0, 0, 1))
                            for sam_img, mask_img in zip(sam_images, mask_images):
                                clothing_image.paste(sam_img, (0, 0), mask=mask_img)
                                mask_image.paste(mask_img, (0, 0), mask=mask_img)

                        # real people
                        if _model_mode == 0:
                            person_boxes, _ = self.dino.dino_predict_internal(_input_image, self.dino_model_name,
                                                                              "person",
                                                                              _box_threshold)
                            # sam_result, person_boxes = self.sam.sam_predict(self.dino_model_name, "person", 0.33, _input_image)
                            if len(person_boxes) == 0:
                                # return {'success': False, 'result': '未检测到服装'}
                                return {'success': False, 'result': 'backend.magic-closet.error.no-cloth'}

                            person0_box = [int(x) for x in person_boxes[0]]
                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            left_ratio = 0.05
                            right_ratio = 0.05
                            top_ratio = 0.2
                            bottom_ratio = 0.2

                            target_left = person0_box[0] - left_ratio * person0_width
                            target_left = 0 if target_left <= 0 else target_left
                            target_top = person0_box[1] - top_ratio * person0_height
                            target_top = 0 if target_top <= 0 else target_top
                            target_right = person0_box[2] + right_ratio * person0_width
                            target_right = _input_image_width if target_right >= _input_image_width else target_right
                            target_bottom = person0_box[3] + bottom_ratio * person0_height
                            target_bottom = _input_image_height if target_bottom >= _input_image_height else target_bottom

                        # artificial model
                        else:
                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            constant_bottom = 40
                            constant_top = 40
                            factor_bottom = 5
                            factor_top = 5
                            left_ratio = 0.35
                            right_ratio = 0.35
                            # top_ratio = 0.32
                            top_ratio = min(0.4, math.pow(person0_width / person0_height, factor_top) * constant_top)
                            bottom_ratio = min(0.6, math.pow(person0_width / person0_height,
                                                             factor_bottom) * constant_bottom)
                            # top_ratio = 0.45
                            # bottom_ratio = 0.6
                            print(
                                f"bottom_ratio1: {math.pow(person0_width / person0_height, factor_bottom) * constant_bottom}")
                            print(f"bottom_ratio: {bottom_ratio}")
                            print(f"top_ratio1: {math.pow(person0_width / person0_height, factor_top) * constant_top}")
                            print(f"top_ratio: {top_ratio}")
                            print(f"boxes: {person0_box}")
                            print(f"width: {person0_width}")
                            print(f"height: {person0_height}")
                            print(f"top increase: {person0_height * top_ratio}")
                            print(f"bottom increase: {person0_height * bottom_ratio}")

                            target_left = person0_box[0] - left_ratio * person0_width
                            target_top = person0_box[1] - top_ratio * person0_height
                            target_top = 0 if person0_box[1] <= 8 else target_top
                            target_right = person0_box[2] + right_ratio * person0_width
                            target_bottom = person0_box[3] + bottom_ratio * person0_height
                            target_bottom = _input_image_height if _input_image_height - person0_box[
                                3] <= 8 else target_bottom

                        target_width = target_right - target_left
                        target_height = target_bottom - target_top
                        resized_clothing_image = self.configure_image(clothing_image,
                                                                      [target_left, target_top, target_right,
                                                                       target_bottom],
                                                                      min_edge=512)
                        resized_input_image = self.configure_image(_input_image, [target_left, target_top, target_right,
                                                                                  target_bottom],
                                                                   min_edge=512)
                        resized_mask_image = self.configure_image(mask_image, [target_left, target_top, target_right,
                                                                               target_bottom],
                                                                  min_edge=512,
                                                                  color=(0, 0, 0))

                        if self.update_progress(30):
                            return {'success': True}
                    except Exception:
                        print(traceback.format_exc())
                        print('preprocess img error')
                    else:
                        sam_result_tmp_png_fp = []
                        for resized_img_type, cache_image in zip(["resized_input", "resized_mask", "resized_clothing"],
                                                                 [
                                                                     resized_input_image if _model_mode == 0 else resized_clothing_image,
                                                                     resized_mask_image,
                                                                     resized_clothing_image]):
                            cache_fp = f"tmp/model_{resized_img_type}_{pic_name}.png"
                            cache_image.save(cache_fp)
                            sam_result_tmp_png_fp.append({'name': cache_fp})

                        else:
                            if self.update_progress(40):
                                return {'success': True}

                # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
                if self.update_progress(50):
                    return {'success': True}

                task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

                sd_model_positive_prompt, sd_model_negative_prompt, ad_face_positive_prompt, sd_bg_positive_prompt, sd_bg_negative_prompt = self.get_prompt(
                    _age, _viewpoint_mode,
                    _model_type,
                    _place_type,
                    _model_mode=_model_mode)

                prompt_styles = None
                init_img = resized_clothing_image

                sketch = None
                init_img_with_mask = None
                inpaint_color_sketch = None
                inpaint_color_sketch_orig = None
                init_img_inpaint = None
                init_mask_inpaint = None
                steps = 30
                sampler_index = 18  # sampling method modules/sd_samplers_kdiffusion.py
                mask_blur = 0
                mask_alpha = 0
                inpainting_fill = 1
                restore_faces = False
                tiling = False
                n_iter = 1
                batch_size = _batch_size
                cfg_scale = 7
                image_cfg_scale = 1.5
                denoising_strength = 1
                seed = -1.0
                subseed = -1.0
                subseed_strength = 0
                seed_resize_from_h = 0
                seed_resize_from_w = 0
                seed_enable_extras = False
                selected_scale_tab = 0
                scale_by = 1
                resize_mode = 2  # 1: crop and resize 2: resize and fill
                inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
                inpaint_full_res_padding = 0
                inpainting_mask_invert = 1  # Mask mode 0: Inpaint masked - 1: Inpaint not masked
                img2img_batch_input_dir = ''
                img2img_batch_output_dir = ''
                img2img_batch_inpaint_mask_dir = ''
                override_settings_texts = []

                # controlnet args
                controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[
                    self.cnet_idx].get_default_ui_unit()
                controlnet_args_unit1.batch_images = ''
                # controlnet_args_unit1.control_mode = 'Balanced' if _model_mode == 0 else 'My prompt is more important'
                controlnet_args_unit1.control_mode = 'ControlNet is more important' if _model_mode == 0 else 'My prompt is more important'
                controlnet_args_unit1.enabled = True
                # controlnet_args_unit1.enabled = _model_mode==0
                controlnet_args_unit1.guidance_end = 1
                controlnet_args_unit1.guidance_start = 0  # ending control step
                controlnet_args_unit1.image = None
                # controlnet_args_unit1.input_mode = batch_hijack.InputMode.SIMPLE
                controlnet_args_unit1.low_vram = False
                # controlnet_args_unit1.model = 'control_v11p_sd15_inpaint'
                # controlnet_args_unit1.module = 'inpaint_only'
                controlnet_args_unit1.model = 'control_v11p_sd15_normalbae'
                controlnet_args_unit1.module = 'normal_bae'
                controlnet_args_unit1.pixel_perfect = True
                controlnet_args_unit1.resize_mode = 'Crop and Resize'
                controlnet_args_unit1.processor_res = 512
                controlnet_args_unit1.threshold_a = 64
                controlnet_args_unit1.threshold_b = 64
                controlnet_args_unit1.weight = 1 if _model_mode == 0 else 0.3
                # controlnet_args_unit1.weight = 0.4 if _model_mode==0 else 0.2
                controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit2.enabled = False
                controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit3.enabled = False

                # adetail
                adetail_enabled = not self.shared.cmd_opts.disable_adetailer
                face_args = {'ad_model': 'face_yolov8n.pt',
                             'ad_prompt': f'{ad_face_positive_prompt}',
                             'ad_negative_prompt': '2 head,poorly drawn face,ugly,cloned face,blurred faces,irregular face',
                             'ad_confidence': 0.3,
                             'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                             'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                             'ad_denoising_strength': 0.4,
                             'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                             'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512,
                             'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
                             'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
                             'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                             'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                             'ad_controlnet_guidance_end': 1,
                             'is_api': ()}
                hand_args = {'ad_model': 'hand_yolov8s.pt',
                             'ad_prompt': 'detailed fingers,realistic fingers,detailed hand,realistic hand',
                             'ad_negative_prompt': 'mutated hands, bad hands, poorly drawn hands,3 hand,3 hand,twisted hands,fused fingers,too many fingers,duplicate,poorly drawn hands,extra fingers',
                             'ad_confidence': 0.3,
                             'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                             'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                             'ad_denoising_strength': 0.4,
                             'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                             'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512,
                             'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
                             'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
                             'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                             'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                             'ad_controlnet_guidance_end': 1,
                             'is_api': ()}
                sam_args = [0,
                            adetail_enabled, face_args, hand_args,  # adetail args
                            controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                            # sam
                            True, False, 0, resized_input_image if _model_mode == 0 else resized_clothing_image,
                            sam_result_tmp_png_fp,
                            0,  # sam_output_chosen_mask
                            False, sam_result_tmp_png_fp, [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                            False, 0, None, None,
                            # tiled diffsuion
                            False, 'MultiDiffusion', False, True,
                            1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                            64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                            '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '',
                            '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                            # tiled_vae
                            False, 256, 48, True, True, True,
                            False
                            ]

                # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
                if self.update_progress(50):
                    return {'success': True}

                ok_img_count = 0
                fuck_img_count = 0
                ok_res = []
                ok_sam_res = []
                sam_bg_tmp_png_fp_list = []
                _output_model_width, _output_model_height = init_img.size
                # while ok_img_count < batch_size:
                # 模特生成
                res = self.img2img.img2img(task_id, 4, sd_model_positive_prompt, sd_model_negative_prompt,
                                           prompt_styles,
                                           init_img,
                                           sketch,
                                           init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                           init_img_inpaint, init_mask_inpaint,
                                           steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                           restore_faces,
                                           tiling,
                                           n_iter, batch_size, cfg_scale, image_cfg_scale,
                                           denoising_strength,
                                           seed,
                                           subseed,
                                           subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                           seed_enable_extras,
                                           selected_scale_tab, _output_model_height, _output_model_width, scale_by,
                                           resize_mode,
                                           inpaint_full_res,
                                           inpaint_full_res_padding, inpainting_mask_invert,
                                           img2img_batch_input_dir,
                                           img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                           override_settings_texts,
                                           *sam_args)

                self.devices.torch_gc()
                for res_idx, res_img in enumerate(res[0]):
                    if getattr(res_img, 'already_saved_as', False):
                        if self.predict_image(res_img.already_saved_as):
                            return {'success': False, 'result': 'backend.generate.error.re-try'}
                            # fuck_img_count += 1
                            # if fuck_img_count > 10:
                            #     # return {'success': False, 'result': "生成失败次数过多"}
                            #     return {'success': False, 'result': 'backend.generate.error.re-try'}
                            #
                            # else:
                            #     print('detect nsfw, retry')
                        else:
                            res_img = res_img.convert('RGBA')
                            # sam
                            sam_bg_result, person_boxes = self.sam.sam_predict(self.dino_model_name, 'person', 0.3,
                                                                               res_img)

                            sam_bg_tmp_png_fp = []
                            if len(sam_bg_result) > 0:
                                for idx, sam_mask_img in enumerate(sam_bg_result):
                                    cache_fp = f"tmp/model_only_person_seg_{res_idx}_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
                                    sam_bg_result[idx].save(cache_fp)
                                    sam_bg_tmp_png_fp.append({'name': cache_fp})
                                else:
                                    sam_bg_tmp_png_fp_list.append(sam_bg_tmp_png_fp)
                                ok_img_count += 1
                                ok_res.append(sam_bg_result[2])
                                ok_sam_res.append(sam_bg_result[2])

                            else:
                                # fuck_img_count += 1
                                # if fuck_img_count > 10:
                                # return {'success': False, 'result': "fatal error"}
                                return {'success': False, 'result': "backend.generate.error.failed"}
                                # else:
                                #     print('detect no person, retry')
                else:
                    del res

                # celery_task.update_state(state='PROGRESS', meta={'progress': 70})
                if self.update_progress(70):
                    return {'success': True}

                    # else:
                # 背景生成
                for ok_idx, ok_model_res in enumerate(ok_res):
                    # cache_fp = f"tmp/model_only_{ok_idx}_{pic_name}_save.png"
                    # ok_model_res.save(cache_fp)

                    task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
                    steps = 20
                    sampler_index = 18  # sampling method modules/sd_samplers_kdiffusion.py
                    inpainting_fill = 1
                    restore_faces = False
                    batch_size = 1
                    resize_mode = 2  # 1: crop and resize 2: resize and fill
                    inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
                    inpaint_full_res_padding = 0
                    inpainting_mask_invert = 1  # Mask mode 0: Inpaint masked - 1: Inpaint not masked
                    cfg_scale = 9

                    # controlnet args
                    controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[
                        self.cnet_idx].get_default_ui_unit()

                    controlnet_args_unit1.batch_images = ''
                    controlnet_args_unit1.control_mode = 'My prompt is more important'
                    controlnet_args_unit1.guidance_end = 1
                    controlnet_args_unit1.guidance_start = 0  # ending control step
                    controlnet_args_unit1.image = None
                    controlnet_args_unit1.low_vram = False
                    controlnet_args_unit1.model = 'control_v11p_sd15_normalbae'
                    controlnet_args_unit1.module = 'normal_bae'
                    controlnet_args_unit1.pixel_perfect = True
                    controlnet_args_unit1.resize_mode = 'Crop and Resize'
                    controlnet_args_unit1.processor_res = 512
                    controlnet_args_unit1.threshold_a = 64
                    controlnet_args_unit1.threshold_b = 64
                    controlnet_args_unit1.weight = 0.4
                    controlnet_args_unit1.enabled = True
                    controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
                    controlnet_args_unit2.enabled = False
                    controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
                    controlnet_args_unit3.enabled = False

                    # adetail
                    adetail_enabled = False

                    sam_args = [0,
                                adetail_enabled, face_args, hand_args,  # adetail args
                                controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                                # sam
                                True, False, 0, ok_model_res,
                                sam_bg_tmp_png_fp_list[ok_idx],
                                0,  # sam_output_chosen_mask
                                False, sam_bg_tmp_png_fp_list[ok_idx], [], False, 0, 1, False, False, 0, None, [], -2,
                                False, [],
                                False, 0, None, None,
                                # tiled diffsuion
                                False, 'MultiDiffusion', False, True,
                                1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                                64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2,
                                -1.0,
                                False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2,
                                '',
                                '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                                False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2,
                                '',
                                '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                                False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                                # tiled_vae
                                False, 256, 48, True, True, True,
                                False
                                ]
                    ok_res[ok_idx] = \
                        self.img2img.img2img(task_id, 4, sd_bg_positive_prompt, sd_bg_negative_prompt, prompt_styles,
                                             ok_sam_res[ok_idx],
                                             sketch,
                                             init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                             init_img_inpaint, init_mask_inpaint,
                                             steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                             restore_faces,
                                             tiling,
                                             n_iter, batch_size, cfg_scale, image_cfg_scale,
                                             denoising_strength,
                                             seed,
                                             subseed,
                                             subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                             seed_enable_extras,
                                             selected_scale_tab, _output_model_height, _output_model_width, scale_by,
                                             resize_mode,
                                             # selected_scale_tab, height, width, scale_by, resize_mode,
                                             inpaint_full_res,
                                             inpaint_full_res_padding, inpainting_mask_invert,
                                             img2img_batch_input_dir,
                                             img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                             override_settings_texts,
                                             *sam_args)[0][0]
                    self.devices.torch_gc()

                # celery_task.update_state(state='PROGRESS', meta={'progress': 90})
                if self.update_progress(90):
                    return {'success': True}
                    #  -------------------------------------------------------------------------------------
                # storage img
                img_urls = []
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_model_dir'], user_id)
                os.makedirs(dir_path, exist_ok=True)

                for ok_idx, ok_img in enumerate(ok_res):
                    img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                    # extra upscaler
                    scales = 1
                    gfpgan_weight = 0
                    codeformer_visibility = 1
                    args = (0, scales, None, None, True, 'None', 'None', 0, gfpgan_weight, codeformer_visibility, 0)
                    self.devices.torch_gc()
                    pp = self.scripts_postprocessing.PostprocessedImage(ok_img.convert("RGB"))
                    self.scripts.scripts_postproc.run(pp, args)

                    pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
                    # 限制缓存10张
                    cache_list = sorted(os.listdir(dir_path))
                    if len(cache_list) > 10:
                        os.remove(os.path.join(dir_path, cache_list[0]))
                else:
                    for img_fn in sorted(os.listdir(dir_path), reverse=True):
                        url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=model"
                        img_urls.append(url_fp)
                    if len(img_urls) < 10:
                        for i in range(10 - len(img_urls)):
                            img_urls.append('')
                # celery_task.update_state(state='PROGRESS', meta={'progress': 95})
                if self.update_progress(95):
                    return {'success': True}
                else:
                    # clear images
                    for cache_img_fp in glob.glob(f'tmp/*{pic_name}*'):
                        if '_save' not in cache_img_fp:
                            os.remove(cache_img_fp)

                return {'success': True, 'result': img_urls}

            elif proceed_mode == 'mirror':
                # debug
                if self.shared.sd_model.sd_checkpoint_info.model_name != 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                    self.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')
                # elif self.shared.sd_model.sd_checkpoint_info.model_name == 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                #     self.shared.change_sd_model('dreamshaper_8')
                origin_image_path = f'tmp/mirror_origin_{pic_name}_save.png'
                _input_image.save(origin_image_path, format='PNG')
                # params: {
                #                 gender_enable: genderEnable?.current?.inputValue,
                #                 age_enable: ageEnable?.current?.inputValue,
                #                 face_expression_enable: faceExpressionEnable?.current?.inputValue,
                #                 eye_size_enable: eyeSizeEnable?.current?.inputValue,
                #                 curly_hair_enable: curlyHairEnable?.current?.inputValue,
                #                 muscle_enable: muscleEnable?.current?.inputValue,
                #                 gender: gender,
                #                 gender_is_elder: genderIsElder,
                #                 gender_sim: genderSimRef?.current?.inputValue,
                #                 age_weight: ageWeightRef?.current?.inputValue,
                #                 age_sim: ageSimRef?.current?.inputValue,
                #                 face_expression: faceExpression,
                #                 face_sim: faceExpressionSimRef?.current?.inputValue,
                #                 eye_size: eyeSizeRef?.current?.inputValue,
                #                 curly_hair: curlyHairRef?.current?.inputValue,
                #                 muscle: muscleRef?.current?.inputValue,
                #                 batch_size: batch_size
                #             }

                person_boxes, _ = self.dino.dino_predict_internal(_input_image, self.dino_model_name, 'person', 0.3)
                if len(person_boxes) == 0:
                    return {'success': False, 'result': 'backend.magic-mirror.error.no-body'}

                if self.update_progress(40):
                    return {'success': True}

                # limit 448
                if min(_input_image_width, _input_image_height) < 512:
                    if _input_image_width < _input_image_height:
                        new_width = 512
                        new_height = int(_input_image_height / _input_image_width * new_width)
                    else:
                        new_height = 512
                        new_width = int(_input_image_width / _input_image_height * new_height)
                    _input_image = _input_image.resize((new_width, new_height))

                _input_image = _input_image.convert('RGBA')
                _input_image_width, _input_image_height = _input_image.size
                batch_size = int(params['batch_size'])
                task_list = [k for k, v in params.items() if isinstance(v, dict) and v['enable']]
                result_images = self.proceed_human_transform(params, task_list[0], batch_size, _input_image)
                if isinstance(result_images, dict):
                    return result_images
                else:
                    result_images = result_images[0]
                    if batch_size > 1:
                        result_images = [x.convert('RGBA') for x in result_images]
                        proceed_task_list = task_list[1:]

                        for batch_idx in range(batch_size):
                            for proceed_idx, proceed_task in enumerate(proceed_task_list):
                                result_images[batch_idx] = \
                                    self.proceed_human_transform(params, proceed_task, 1, result_images[batch_idx])[0][
                                        0].convert(
                                        "RGBA")
                                if isinstance(result_images, dict):
                                    return result_images
                                if self.update_progress((batch_idx + 1) * (proceed_idx + 1) * (
                                        80 // (batch_size * len(task_list)))):
                                    return {'success': True}
                    else:
                        if self.update_progress(80):
                            return {'success': True}

                    # storage img
                    img_urls = []
                    dir_path = os.path.join(CONFIG['storage_dirpath']['user_mirror_dir'], user_id)
                    os.makedirs(dir_path, exist_ok=True)
                    for res_idx, res_img in enumerate(result_images):
                        img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                        if 'face_expression' in task_list or 'age' in task_list or 'gender' in task_list:
                            # extra upscaler
                            scales = 1
                            gfpgan_weight = 0
                            codeformer_visibility = 0.5
                            args = (
                                0, scales, None, None, True, 'None', 'None', 0, gfpgan_weight, codeformer_visibility, 0)
                            pp = self.scripts_postprocessing.PostprocessedImage(res_img.convert("RGB"))
                            self.scripts.scripts_postproc.run(pp, args)
                            self.devices.torch_gc()

                            pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=80, lossless=True)
                        else:
                            res_img.convert("RGB").save(os.path.join(dir_path, img_fn), format="jpeg", quality=80,
                                                        lossless=True)

                        # 限制缓存10张
                        cache_list = sorted(os.listdir(dir_path))
                        if len(cache_list) > 10:
                            os.remove(os.path.join(dir_path, cache_list[0]))
                    else:
                        for img_fn in sorted(os.listdir(dir_path), reverse=True):
                            url_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category=mirror"
                            img_urls.append(url_fp)
                        if len(img_urls) < 10:
                            for i in range(10 - len(img_urls)):
                                img_urls.append('')

                    # celery_task.update_state(state='PROGRESS', meta={'progress': 95})
                    if self.update_progress(99):
                        return {'success': True}

                    return {'success': True, 'result': img_urls}
            else:
                pattern = re.compile(r"user_(.*?)_history")
                match = pattern.search(kwargs['input_image'])

                if match:
                    _input_image_mode = match.group(1)
                else:
                    _input_image_mode = 'facer'
                # hires
                if _input_image_mode == 'avatar' or _input_image_mode == 'mirage':
                    if self.shared.sd_model.sd_checkpoint_info.model_name != 'dreamshaper_8':
                        self.shared.change_sd_model('dreamshaper_8')
                elif _input_image_mode == 'facer':
                    pass
                else:
                    if self.shared.sd_model.sd_checkpoint_info.model_name != 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                        self.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')

                # _input_image = base64_to_pil(params['input_image'])
                _output_width = int(params['output_width'])
                _output_height = int(params['output_height'])

                _output_ratio = _output_width / _output_height
                _input_ratio = _input_image_width / _input_image_height

                # celery_task.update_state(state='PROGRESS', meta={'progress': 10})
                if self.update_progress(10):
                    return {'success': True}
                if _input_ratio != _output_ratio:
                    padding_height = int(
                        _input_image_width / _output_ratio) if _input_image_width <= _input_image_height else _input_image_height
                    padding_width = _input_image_width if _input_image_width <= _input_image_height else int(
                        _input_image_height * _output_ratio)

                    padding_height = padding_height // 8 * 8
                    padding_width = padding_width // 8 * 8

                    # img2img padding to _output_ratio
                    steps = 20
                    sampler_index = 18  # sampling method modules/sd_samplers_kdiffusion.py

                    task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
                    sd_positive_prompt = '<lora:polyhedron_new_skin_v1.1:0.1>,(best quality:1.2),(high quality:1.2),masterpiece,high details,(Realism:1.4), vivid color, (realistic, photo-realistic:1.3), masterpiece'
                    sd_negative_prompt = '(NSFW:1.8),paintings, sketches, (worst quality:2), (low quality:2), lowres, ((monochrome)), ((grayscale))'
                    prompt_styles = None
                    init_img = _input_image
                    sketch = None
                    init_img_with_mask = None
                    inpaint_color_sketch = None
                    inpaint_color_sketch_orig = None
                    init_img_inpaint = None
                    init_mask_inpaint = None
                    mask_blur = 0
                    mask_alpha = 0
                    inpainting_fill = 1
                    restore_faces = False
                    tiling = False
                    n_iter = 1
                    batch_size = 1
                    cfg_scale = 7
                    image_cfg_scale = 1.5
                    denoising_strength = 1
                    seed = -1.0
                    subseed = -1.0
                    subseed_strength = 0
                    seed_resize_from_h = 0
                    seed_resize_from_w = 0
                    seed_enable_extras = False
                    selected_scale_tab = 0
                    scale_by = 1
                    resize_mode = 2
                    inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
                    inpaint_full_res_padding = 32
                    inpainting_mask_invert = 0
                    img2img_batch_input_dir = ''
                    img2img_batch_output_dir = ''
                    img2img_batch_inpaint_mask_dir = ''
                    override_settings_texts = []

                    # controlnet args
                    controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[
                        self.cnet_idx].get_default_ui_unit()
                    controlnet_args_unit1.batch_images = ''
                    controlnet_args_unit1.control_mode = 'ControlNet is more important'
                    controlnet_args_unit1.enabled = True
                    controlnet_args_unit1.guidance_end = 1
                    controlnet_args_unit1.guidance_start = 0  # ending control step
                    controlnet_args_unit1.image = {'image': np.array(_input_image.convert('RGB')),
                                                   'mask': np.full((_input_image_height, _input_image_width, 4),
                                                                   (0, 0, 0, 255), dtype=np.uint8)}
                    controlnet_args_unit1.low_vram = False
                    controlnet_args_unit1.model = 'control_v11p_sd15_inpaint'
                    controlnet_args_unit1.module = 'inpaint_only+lama'
                    controlnet_args_unit1.pixel_perfect = True
                    controlnet_args_unit1.resize_mode = 'Resize and Fill'
                    controlnet_args_unit1.weight = 1
                    controlnet_args_unit1.threshold_a = 64
                    controlnet_args_unit1.threshold_b = 64
                    controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
                    controlnet_args_unit2.enabled = False
                    controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
                    controlnet_args_unit3.enabled = False

                    # adetail
                    adetail_enabled = False
                    fake_args = {'ad_model': 'face_yolov8m.pt', 'ad_prompt': '', 'ad_negative_prompt': '',
                                 'ad_confidence': 0.3,
                                 'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                                 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 0,
                                 'ad_denoising_strength': 0.4,
                                 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                                 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
                                 'ad_inpaint_height': 512,
                                 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
                                 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
                                 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                                 'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0,
                                 'ad_controlnet_guidance_end': 1,
                                 'is_api': ()}
                    sam_args = [0,
                                adetail_enabled, fake_args, fake_args,  # adetail args
                                controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                                False, False, 0, None,  # sam args
                                [], 0, False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [], False, 0,
                                None, None,
                                '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '',
                                False, 'MultiDiffusion', False,
                                True, 1024, 1024, 64, 64, 32, 8, 'None', 2, False, 10, 1, 1,
                                64, False, False, False, False, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2,
                                -1.0,
                                False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2,
                                '',
                                '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                                False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2,
                                '',
                                '', 'Background', 0.2, -1.0, False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                                False, 0.4, 0.4, 0.2, 0.2, '', '', 'Background', 0.2, -1.0,
                                # tiled_vae
                                False, 256, 48, True, True, True,
                                False
                                ]

                    cnet_res = self.img2img.img2img(task_id, 0, sd_positive_prompt, sd_negative_prompt,
                                                    prompt_styles, init_img,
                                                    sketch,
                                                    init_img_with_mask, inpaint_color_sketch,
                                                    inpaint_color_sketch_orig,
                                                    init_img_inpaint, init_mask_inpaint,
                                                    steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                                    restore_faces,
                                                    tiling,
                                                    n_iter, batch_size, cfg_scale, image_cfg_scale,
                                                    denoising_strength, seed,
                                                    subseed,
                                                    subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                                    seed_enable_extras,
                                                    selected_scale_tab, padding_height, padding_width, scale_by,
                                                    resize_mode,
                                                    inpaint_full_res,
                                                    inpaint_full_res_padding, inpainting_mask_invert,
                                                    img2img_batch_input_dir,
                                                    img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                                    override_settings_texts,
                                                    *sam_args)
                else:
                    padding_height = _input_image_height // 8 * 8
                    padding_width = _input_image_width // 8 * 8
                # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
                if self.update_progress(50):
                    return {'success': True}
                self.devices.torch_gc()
                # cnet_res[0][0].save(f'tmp/cnet_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                #                   format='PNG')
                # extra upscaler
                cnet_res_img = _input_image if _output_ratio == _input_ratio else cnet_res[0][0]
                scales = _output_width / padding_width

                # celery_task.update_state(state='PROGRESS', meta={'progress': 70})
                if self.update_progress(70):
                    return {'success': True}

                gfpgan_weight = 0
                codeformer_visibility = 1 if _input_image_mode == 'model' else 0
                args = (0, scales, None, None, True, 'R-ESRGAN 4x+', 'None', 0, gfpgan_weight, codeformer_visibility,
                        0 if _input_image_mode == 'model' else 1)
                assert cnet_res_img, 'image not selected'
                self.devices.torch_gc()
                pp = self.scripts_postprocessing.PostprocessedImage(cnet_res_img.convert("RGB"))
                self.scripts.scripts_postproc.run(pp, args)

                self.devices.torch_gc()

                # celery_task.update_state(state='PROGRESS', meta={'progress': 80})
                if self.update_progress(80):
                    return {'success': True}

                dir_path = CONFIG['storage_dirpath']['hires_dir']
                os.makedirs(dir_path, exist_ok=True)

                img_fn = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{''.join([random.choice(string.ascii_letters) for c in range(6)])}.jpeg"
                img_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}"
                # pp.image.save(os.path.join(dir_path, img_fn), format="png", quality=100)
                pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=100, lossless=True)
                # celery_task.update_state(state='PROGRESS', meta={'progress': 90})
                if self.update_progress(90):
                    return {'success': True}
                return {'success': True, 'result': [img_fp]}

        except Exception:
            print('errrrr!!!!!!!!!!!!!!')
            self.logging(
                f"[predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        return {'success': False, 'result': 'backend.generate.error.failed'}

# if __name__ == '__main__':
#     op = OperatorSD()
#     op()
