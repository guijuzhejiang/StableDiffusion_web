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
import io
import importlib
import json
import os
import sys
from collections import OrderedDict

import GPUtil
from PIL import Image
import copy
import datetime
import math
import random
import string
import traceback
import ujson
import cv2
import numpy as np

from lora_config import lora_model_dict, lora_gender_dict, lora_model_common_dict, lora_place_dict
from lib.celery_workshop.operator import Operator
from utils.global_vars import CONFIG



class OperatorSD(Operator):
    """ stable diffusion """
    num = len(GPUtil.getGPUs())
    cache = True
    cuda = True
    enable = True
    celery_task_name = 'sd_task'

    def __init__(self, gpu_idx=0):
        os.environ['ACCELERATE'] = 'True'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        print("use gpu:" + str(gpu_idx))
        Operator.__init__(self)
        self.extra_networks = importlib.import_module('modules.extra_networks')
        self.script_callbacks = importlib.import_module('modules.script_callbacks')
        self.dino = importlib.import_module('guiju.segment_anything_util.dino')
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
        # self.sam_predict = importlib.import_module('guiju.segment_anything_util.sam.sam_predict')
        self.predict_image = getattr(importlib.import_module('guiju.predictor_opennsfw2'), 'predict_image')
        self.logging = getattr(importlib.import_module('lib.common.common_util'), 'logging')
        self.logging = getattr(importlib.import_module('lib.common.common_util'), 'logging')
        self.img2img = getattr(importlib.import_module('modules'), 'img2img')
        self.devices = getattr(importlib.import_module('modules'), 'devices')
        self.scripts_postprocessing = getattr(importlib.import_module('modules'), 'scripts_postprocessing')

        os.makedirs(CONFIG['storage_dirpath']['user_dir'], exist_ok=True)

        # self.lora_model_dict = {}
        # for fn in os.listdir(CONFIG['storage_dirpath']['lora_model_dir']):
        #     index, lora_text = fn.split('_')
        #     lora_name, weight = lora_text.split(':')
        #     self.lora_model_dict[index] = {'name': lora_name, 'weight': weight}
        # self.lora_place_dict = {}
        # for fn in os.listdir(CONFIG['storage_dirpath']['lora_place_dir']):
        #     index, lora_text = fn.split('_')
        #     lora_name, weight = lora_text.split(':')
        #     self.lora_model_dict[index] = {'name': lora_name, 'weight': weight}

        self.shared.cmd_opts.gradio_debug = True
        self.shared.cmd_opts.listen = True
        self.shared.cmd_opts.debug_mode = True
        self.shared.cmd_opts.enable_insecure_extension_access = True
        self.shared.cmd_opts.xformers = True
        self.shared.cmd_opts.disable_tls_verify = True
        self.shared.cmd_opts.hide_ui_dir_config = True
        self.shared.cmd_opts.disable_tome = False
        self.shared.cmd_opts.lang = 'ch'
        self.shared.cmd_opts.disable_adetailer = False
        self.initialize()
        self.sam.sam = self.sam.init_sam_model()
        dino_model, dino_name = self.dino.load_dino_model2("GroundingDINO_SwinB (938MB)")
        self.dino.dino_model_cache[dino_name] = dino_model
        # ui_extra_networks.initialize()
        # ui_extra_networks.register_default_pages()

        # extra_networks.initialize()
        # extra_networks.register_default_extra_networks()
        # modules.script_callbacks.before_ui_callback()

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

        cnet_idx = 1
        sam_idx = 2
        adetail_idx = 0
        self.scripts.scripts_img2img.alwayson_scripts[0], \
        self.scripts.scripts_img2img.alwayson_scripts[1], \
        self.scripts.scripts_img2img.alwayson_scripts[2] \
            = self.scripts.scripts_img2img.alwayson_scripts[sam_idx], \
              self.scripts.scripts_img2img.alwayson_scripts[cnet_idx], \
              self.scripts.scripts_img2img.alwayson_scripts[adetail_idx]

        # sam 24 args
        self.scripts.scripts_img2img.alwayson_scripts[0].args_from = 7
        self.scripts.scripts_img2img.alwayson_scripts[0].args_to = 31

        # controlnet 3 args
        self.scripts.scripts_img2img.alwayson_scripts[1].args_from = 4
        self.scripts.scripts_img2img.alwayson_scripts[1].args_to = 7

        # adetail 3 args
        self.scripts.scripts_img2img.alwayson_scripts[2].args_from = 1
        self.scripts.scripts_img2img.alwayson_scripts[2].args_to = 4

        # invisible detectmap
        self.shared.opts.control_net_no_detectmap = True

        print('init done')

    def configure_image(self, image, person_pos, target_ratio=0.5, quality=90, padding=8):
        person_pos = [int(x) for x in person_pos]
        # 将PIL RGBA图像转换为BGR图像
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

        # 获取原始图像的尺寸
        original_height, original_width = cv_image.shape[:2]

        # 计算模特图像的长宽比
        person_height = person_pos[3] - person_pos[1]
        person_width = person_pos[2] - person_pos[0]
        person_ratio = person_width / person_height

        # 计算应该添加的填充量
        if person_ratio > target_ratio:
            # 需要添加垂直box
            target_height = int(person_width / target_ratio)
            remainning_height = original_height - target_height
            if remainning_height >= 0:
                top = int((target_height - person_height) / 2)
                bottom = target_height - person_height - top

                if person_pos[1] - top < 0:
                    padded_image = cv_image[:person_pos[3] + bottom - person_pos[1] + top, person_pos[0]:person_pos[2]]

                else:
                    padded_image = cv_image[person_pos[1] - top:person_pos[3] + bottom, person_pos[0]:person_pos[2]]

            else:
                # top = int((target_height - original_height) / 2)
                # bottom = target_height - original_height - top
                # padded_image = cv2.copyMakeBorder(cv_image[padding:original_height-padding, :], top+padding, bottom+padding, 0, 0, cv2.BORDER_REPLICATE)
                # padded_image = padded_image[:, person_pos[0]:person_pos[2]]
                padded_image = cv_image
        else:
            # 需要添加水平box
            target_width = int(person_height * target_ratio)
            remainning_width = original_width - target_width
            if remainning_width >= 0:
                left = int((target_width - person_width) / 2)
                right = target_width - person_width - left

                if person_pos[0] - left < 0:
                    padded_image = cv_image[person_pos[1]:person_pos[3], :person_pos[2] + right - person_pos[0] + left]

                else:
                    padded_image = cv_image[person_pos[1]:person_pos[3], person_pos[0] - left:person_pos[2] + right]
            else:
                # left = int((target_width - original_width) / 2)
                # right = target_width - original_width - left
                # padded_image = cv2.copyMakeBorder(cv_image[:, padding:original_width-padding], 0, 0, left+padding, right+padding, cv2.BORDER_REPLICATE)
                # padded_image = padded_image[person_pos[1]:person_pos[3], :]
                padded_image = cv_image

        padded_image = cv2.cvtColor(np.array(padded_image), cv2.COLOR_BGRA2RGBA)
        return padded_image

    def padding_rgba_image_pil_to_cv(self, original_image, pl, pr, pt, pb, person_box, padding=8):
        original_width, original_height = original_image.size
        edge_color = original_image.getpixel((padding, padding))
        # padded_image = Image.new('RGBA', (original_width + pl + pr, original_height + pt + pb), edge_color)
        # padded_image.paste(original_image, (pl, pt), mask=original_image)

        cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGBA2BGRA)
        h, w, _ = cv_image.shape

        padded_image = cv2.copyMakeBorder(cv_image[padding:h - padding, padding:w - padding],
                                          pt + padding if person_box[1] > 8 else 0,
                                          pb + padding if 8 <= original_height - person_box[3] else 0,
                                          pl + padding if person_box[0] > 8 else 0,
                                          pr + padding if 8 <= original_width - person_box[2] else 0,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded_image = cv2.copyMakeBorder(padded_image,
                                          pt + padding if person_box[1] <= 8 else 0,
                                          pb + padding if 8 > original_height - person_box[3] else 0,
                                          pl + padding if person_box[0] <= 8 else 0,
                                          pr + padding if 8 > original_width - person_box[2] else 0,
                                          cv2.BORDER_CONSTANT,
                                          value=(255, 255, 255)
                                          )
        padded_image = cv2.cvtColor(np.array(padded_image), cv2.COLOR_BGRA2RGBA)
        return padded_image

    def get_prompt(self, _age, _viewpoint, _model_type, _place_type, _model_mode=0):
        sd_positive_prompts_dict = OrderedDict({
            # 'gender': [
            #     # female
            #     '1girl',
            #     # male
            #     '1boy',
            # ],
            'age': [
                # child
                f'(child:1.3)',
                # youth
                # f'(youth:1.3){"" if _gender else ", <lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>"}',
                f'(youth:1.3)',
                # middlescent
                '(middlescent:1.3)',
            ],
            'common': [
                '(full body:1.3)',
                '(best quality:1.2)',
                '(high quality:1.2)',
                'high details',
                '(Realism:1.4)',
                'masterpiece',
                'extremely detailed,extremely delicate,Amazing,8k wallpaper',
                'long legs, correct body proportions, good figure',
                'detailed fingers',
                'realistic fingers',
                'detailed hand',
                'realistic hand',
                # 'detailed foot',
                'realistic body',
                'out of frame',
                '' if _viewpoint == 2 else 'posing for a photo, realistic face',
                # '(simple background:1.3)',
                # '(white background:1.3)',
            ],
            'viewpoint': [
                # 正面
                'light smile,looking at viewer,beautiful detailed face,beautiful detailed nose,beautiful detailed eyes',
                # 侧面
                'light smile, a side portrait photo of a people, (looking to the side:1.5)',
                # 反面
                '(a person with their back to the camera:1.5)'
            ]
        })

        sd_positive_prompts_dict['common'] = [x for x in sd_positive_prompts_dict['common'] if x]
        sd_positive_prompts_dict['age'] = [sd_positive_prompts_dict['age'][_age]]
        sd_positive_prompts_dict['viewpoint'] = [sd_positive_prompts_dict['viewpoint'][_viewpoint]]

        if _viewpoint == 2:
            # sd_positive_prompt = f'(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, 2k wallpaper,realistic body, (simple background:1.3), (white background:1.3), (from behind:1.3){", 1boy" if _gender else ""}'
            sd_positive_prompt = f'RAW photo, best quality, realistic, photo-realistic:1.3, masterpiece, 2k wallpaper,(from behind:1.3)'

        else:
            sd_positive_prompt = ', '.join([i for x in sd_positive_prompts_dict.values() for i in x])

        sd_negative_prompt = f'{"" if _place_type ==0 else "(plain background:1.3), (simple background:1.3), (white background:1.3),"}(overexposure:1.5),(exposure:1.5), (extra clothes:1.5),(clothes:1.5),(NSFW:1.3),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), clothing, pants, shorts, t-shirt, dress, sleeves, lowres, ((monochrome)), ((grayscale)), duplicate, morbid, mutilated, mutated hands, poorly drawn face,skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, too many fingers, long neck, cross-eyed, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8), aged up, old fingers, long neck, cross-eyed, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, bad body, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8)'

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

            ad_face_positive_prompt = ', '.join(lora_prompt_list)

        for lora_common in lora_model_common_dict:
            lora_prompt_list.append(f"<lora:{lora_common['lora_name']}:{lora_common['weight']}>")
        else:
            lora_prompt_list.append(lora_place_dict[_place_type]['prompt'])

        sd_positive_prompt = f"{sd_positive_prompt}, {', '.join(lora_prompt_list)}"

        print(f'sd_positive_prompt: {sd_positive_prompt}')
        print(f'sd_negative_prompt: {sd_negative_prompt}')
        return sd_positive_prompt, sd_negative_prompt, ad_face_positive_prompt

    def __call__(self, *args, **kwargs):
        try:
            # init lora
            self.extra_networks.initialize()
            self.extra_networks.register_default_extra_networks()
            self.script_callbacks.before_ui_callback()
            print("operation start !!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(args)
            print({k:v for k, v in kwargs.items() if k != 'input_image'})
            proceed_mode = kwargs['mode'][0]

            if proceed_mode == 'model':
                params = ujson.loads(kwargs['params'][0])
                _cloth_part = 0
                _batch_size = int(params['batch_size'])
                # _input_image = base64_to_pil(params['input_image'])
                _input_image = Image.open(io.BytesIO(kwargs['input_image'][0][1]))
                # _gender = 0 if params['gender'] == 'female' else 1
                arge_idxs = {v: i for i, v in enumerate(['child', 'youth', 'middlescent'])}
                _age = arge_idxs[params['age']]
                viewpoint_mode_idxs = {v: i for i, v in enumerate(['front', 'side', 'back'])}
                _viewpoint_mode = viewpoint_mode_idxs[params['viewpoint_mode']]
                _model_mode = 0 if params['model_mode'] == 'normal' else 1
                # model and place type
                _model_type = int(params['model_type'])
                _place_type = int(params['place_type'])

                _output_height = 768
                _output_width = 512

                _dino_model_name = "GroundingDINO_SwinB (938MB)"
                # _sam_model_name = 'samhq_vit_h_1b3123.pth'

                # _dino_clothing_text_prompt = 'clothing . pants . shorts . dress . shirt . t-shirt . skirt . underwear . bra . swimsuits . bikini . stocking . chain . bow' if _model_mode == 1 else 'clothing . pants . shorts'
                # _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear . bra . swimsuit . bikini . stocking . chain . bow'
                # _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear . bra . bikini . bowtie'
                #underwear . bikini和bowtie冲突，bra和bowtie不冲突，考虑分组遍历后在合并
                # _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . bra . bowtie'
                #bikini和t-shirt冲突
                # _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear'
                _dino_clothing_text_prompt = [
                    'clothing . pants . dress . shirt . t-shirt',
                    'bra . bikini . bowtie . stocking . chain . underwear',
                ]
                # _dino_clothing_text_prompt_0 = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear'
                # _dino_clothing_text_prompt_1 = 'bra . bikini . bowtie . stocking . chain'
                _box_threshold = 0.3

                if _input_image is None:
                    return None, None
                else:
                    origin_image_path = f'tmp/origin_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
                    _input_image.save(origin_image_path, format='PNG')

                    try:
                        if self.predict_image(origin_image_path):
                            return {'success': False, 'result': "抱歉，您上传的图像未通过合规性检查，请重新上传。"}

                        _input_image_width, _input_image_height = _input_image.size

                        # real people
                        if _model_mode == 0:
                            person_boxes, _ = self.dino.dino_predict_internal(_input_image, _dino_model_name, "person",
                                                                    _box_threshold)
                            if len(person_boxes) == 0:
                                return {'success': False, 'result': '未检测到服装'}

                            person0_box = [int(x) for x in person_boxes[0]]
                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            top_ratio = 0.2
                            bottom_ratio = 0.2
                            if (person0_box[1] / person0_height) <= top_ratio:
                                person0_box[1] = 0
                            else:
                                person0_box[1] -= top_ratio * person0_height

                            if (_input_image_height - person0_box[3]) / person0_height <= bottom_ratio:
                                person0_box[3] = _input_image_height
                            else:
                                person0_box[3] += bottom_ratio * person0_height
                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            _input_image = self.configure_image(_input_image, person0_box, target_ratio=_output_width / _output_height if (person0_width / person0_height) < (_output_width / _output_height) else person0_width / person0_height)

                            if self.shared.cmd_opts.debug_mode:
                                cv2.imwrite(f'tmp/person_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                                            cv2.cvtColor(np.array(_input_image), cv2.COLOR_RGBA2BGRA))
                        # artificial model
                        else:
                            person0_box = [-1, -1, -1, -1]
                            for dino_prompt in _dino_clothing_text_prompt:
                                person_boxes, _ = self.dino.dino_predict_internal(_input_image, _dino_model_name,
                                                                    dino_prompt, _box_threshold)
                                if len(person_boxes) > 0:
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

                            if person0_box[0] == -1:
                                return {'success': False, 'result': '未检测到服装'}

                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            constant_bottom = 40
                            constant_top = 40
                            factor_bottom = 4
                            factor_top = 4
                            left_ratio = 0.1
                            right_ratio = 0.1
                            # top_ratio = 0.32
                            top_ratio = min(0.35, math.pow(person0_width / person0_height, factor_top) * constant_top)
                            bottom_ratio = min(0.58, math.pow(person0_width / person0_height, factor_bottom) * constant_bottom)
                            print(f"bottom_ratio: {bottom_ratio}")
                            print(f"top_ratio: {top_ratio}")
                            print(f"boxes: {person_boxes}")
                            print(f"width: {person0_width}")
                            print(f"height: {person0_height}")
                            print(f"top increase: {person0_height * top_ratio}")
                            print(f"bottom increase: {person0_height * bottom_ratio}")

                            # padding_left = int(person0_width * left_ratio - int(person0_box[0])) if (int(person0_box[0]) / person0_width) < left_ratio else 0
                            # padding_right = int(person0_width * right_ratio - (_input_image_width - int(person0_box[2]))) if ((_input_image_width - int(person0_box[2])) / person0_width) < right_ratio else 0
                            padding_left = 0
                            padding_right = 0
                            padding_top = int(person0_height * top_ratio - int(person0_box[1])) if (int(
                                person0_box[1]) / person0_height) < top_ratio else 0
                            padding_bottom = int(
                                person0_height * bottom_ratio - (_input_image_height - int(person0_box[3]))) if ((
                                                                                                                         _input_image_height - int(
                                                                                                                     person0_box[
                                                                                                                         3])) / person0_height) < bottom_ratio else 0

                            _input_image = self.padding_rgba_image_pil_to_cv(_input_image, padding_left, padding_right,
                                                                             padding_top, padding_bottom, person0_box)

                            _input_image = self.configure_image(_input_image,
                                                                [0 if (int(
                                                                    person0_box[0]) / person0_width) < left_ratio else
                                                                 person0_box[0] - int(person0_width * left_ratio),
                                                                 padding_top if padding_top > 0 else person0_box[
                                                                                                         1] - int(
                                                                     person0_height * top_ratio),
                                                                 _input_image_width if ((_input_image_width - int(
                                                                     person0_box[2])) / person0_width) < right_ratio else padding_left +
                                                                                                                  person0_box[
                                                                                                                      2] + int(
                                                                     person0_width * right_ratio),
                                                                 _input_image_height + padding_bottom + padding_top if padding_bottom > 0 else padding_top +
                                                                                                                                               person0_box[
                                                                                                                                                   3] + int(
                                                                     person0_height * bottom_ratio)],
                                                                target_ratio=_output_width / _output_height if (person0_width / person0_height) < (_output_width / _output_height) else person0_width / person0_height)

                    except Exception:
                        print(traceback.format_exc())
                        print('preprocess img error')
                    else:
                        # # 压缩图像质量
                        quality = 80
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                        _, jpeg_data = cv2.imencode('.jpg', cv2.cvtColor(np.array(_input_image), cv2.COLOR_RGBA2BGRA),
                                                    encode_param)

                        # 将压缩后的图像转换为PIL图像
                        _input_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')

                        # limit height 768
                        check_w, check_h = _input_image.size
                        print(f"before:{_input_image.size}")

                        if check_h > _output_height:
                            tmp_w = int(_output_height * check_w / check_h)
                            tmp_w = int(tmp_w // 8 * 8)
                            _input_image = _input_image.resize((tmp_w, _output_height))

                        else:
                            tmp_h = int(check_h // 8 * 8)
                            tmp_w = int(check_w // 8 * 8)
                            _input_image = _input_image.crop((0, 0, tmp_w, tmp_h))

                    if self.shared.cmd_opts.debug_mode:
                        _input_image.save(f'tmp/resized_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                                          format='PNG')

                sam_result_tmp_png_fp = []
                sam_result_gallery = [None, None, None]
                sam_mask_result = []
                for dino_prompt in _dino_clothing_text_prompt:
                    sam_result, _ = self.sam.sam_predict(_dino_model_name, dino_prompt,
                                                                 _box_threshold,
                                                                 _input_image)
                    if len(sam_result) > 0:
                        if sam_result_gallery[0] is None:
                            sam_result_gallery[0] = sam_result[0]
                            sam_result_gallery[1] = 1
                            sam_result_gallery[2] = sam_result[2]
                        else:
                            sam_result_gallery[0].paste(sam_result[0], (0, 0), sam_result[0])
                            sam_result_gallery[1] = 1
                            sam_result_gallery[2].paste(sam_result[2], (0, 0), sam_result[2])
                        sam_mask_result.append(np.array(sam_result[1]))

                if sam_result_gallery[0] is None:
                    return {'success': False, 'result': '未检测到服装'}
                else:
                    pic_name = ''.join([random.choice(string.ascii_letters) for c in range(15)])

                    merged_mask = None
                    for idx, mask_res in enumerate(sam_mask_result):
                        Image.fromarray(mask_res).save(f'tmp/{idx}_mask_{pic_name}.png')
                        if merged_mask is None:
                            merged_mask = mask_res
                        else:
                            merged_mask |= mask_res
                    else:
                        sam_result_gallery[1] = Image.fromarray(merged_mask)

                for idx, sam_mask_img in enumerate(sam_result_gallery):
                    cache_fp = f"tmp/{idx}_{pic_name}.png"
                    sam_mask_img.save(cache_fp)
                    sam_result_tmp_png_fp.append({'name': cache_fp})

                task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

                sd_positive_prompt, sd_negative_prompt, ad_face_positive_prompt = self.get_prompt(_age, _viewpoint_mode,
                                                                                                  _model_type,
                                                                                                  _place_type,
                                                                                                  _model_mode=_model_mode)

                prompt_styles = None
                _input_image_width, _input_image_height = _input_image.size
                # new_img = Image.open("/home/zzg/workspace/pycharm/StableDiffusion_web/outputs/img2img-images/web_storage/20230908154917_GmdUoW.png").resize((_input_image_width, _input_image_height))
                # new_img.paste(sam_result_gallery[2], (0, 0), sam_result_gallery[2])

                # init_img = new_img
                # init_img = Image.alpha_composite(sam_result_gallery[2], Image.new("RGBA", (_input_image_width, _input_image_height), (55, 55, 55)))
                # init_img = _input_image
                # for x in range(_input_image_width):
                #     for y in range(_input_image_height):
                #         pixel = sam_result_gallery[2].getpixel((x, y))
                #         if pixel[3] == 0:  # 如果透明
                #             # 随机生成噪点的RGB值
                #             noise_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                #             sam_result_gallery[2].putpixel((x, y), noise_color)
                init_img = sam_result_gallery[2]
                # init_img.save('test!!initimg.png')

                sketch = None
                init_img_with_mask = None
                inpaint_color_sketch = None
                inpaint_color_sketch_orig = None
                init_img_inpaint = None
                init_mask_inpaint = None
                steps = 20
                sampler_index = 18  # sampling method modules/sd_samplers_kdiffusion.py
                mask_blur = 4
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
                resize_mode = 2 # 1: crop and resize 2: resize and fill
                inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
                inpaint_full_res_padding = 0
                inpainting_mask_invert = 1 # Mask mode 0: Inpaint masked - 1: Inpaint not masked
                img2img_batch_input_dir = ''
                img2img_batch_output_dir = ''
                img2img_batch_inpaint_mask_dir = ''
                override_settings_texts = []

                # controlnet args
                cnet_idx = 1
                controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[cnet_idx].get_default_ui_unit()
                controlnet_args_unit1.batch_images = ''
                # controlnet_args_unit1.control_mode = 'Balanced' if _model_mode == 0 else 'My prompt is more important'
                controlnet_args_unit1.control_mode = 'My prompt is more important'
                # controlnet_args_unit1.enabled = True
                controlnet_args_unit1.enabled = (_place_type == 0)
                # controlnet_args_unit1.enabled = (_place_type == 0 and _model_mode == 0)
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
                controlnet_args_unit1.weight = 0.8
                controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit2.enabled = False
                controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
                controlnet_args_unit3.enabled = False

                # sam
                # sam_args = [0, True, False, 0, _input_image,
                #             sam_result_tmp_png_fp,
                #             0,  # sam_output_chosen_mask
                #             False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                #             '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                #             True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                #             f'<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: {denoising_strength}</p>',
                #             128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'],
                #             False, False, 'positive', 'comma', 0, False, False, '',
                #             '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                #             64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0
                #             ]

                # adetail
                adetail_enabled = not self.shared.cmd_opts.disable_adetailer
                face_args = {'ad_model': 'face_yolov8m.pt',
                             'ad_prompt': f'{ad_face_positive_prompt},beautiful detailed nose,beautiful detailed eyes',
                             'ad_negative_prompt': '2 head, poorly drawn face, ugly, cloned face, blurred faces, irregular face',
                             'ad_confidence': 0.3,
                             'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                             'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4,
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
                             'ad_prompt': 'detailed fingers,realistic fingers,detailed hand, realistic hand',
                             'ad_negative_prompt': 'mutated hands, bad hands, poorly drawn hands, 3 hand, 3 hand, twisted hands, fused fingers, too many fingers, duplicate, poorly drawn hands, extra fingers',
                             'ad_confidence': 0.3,
                             'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                             'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4,
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
                            True, False, 0, _input_image,
                            sam_result_tmp_png_fp,
                            0,  # sam_output_chosen_mask
                            False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                            '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                            True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                            f'<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: {denoising_strength}</p>',
                            128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0,
                            ['left', 'right', 'up', 'down'],
                            False, False, 'positive', 'comma', 0, False, False, '',
                            '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                            64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None, False,
                            None, None,
                            False, None, None, False, 50
                            ]

                ok_img_count = 0
                fuck_img_count = 0
                ok_res = []
                while ok_img_count < batch_size:
                    res = self.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt, prompt_styles,
                                                  init_img,
                                                  sketch,
                                                  init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                                  init_img_inpaint, init_mask_inpaint,
                                                  steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                                  restore_faces,
                                                  tiling,
                                                  n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength,
                                                  seed,
                                                  subseed,
                                                  subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                                  seed_enable_extras,
                                                  selected_scale_tab, _output_height, _output_width, scale_by, resize_mode,
                                                  # selected_scale_tab, height, width, scale_by, resize_mode,
                                                  inpaint_full_res,
                                                  inpaint_full_res_padding, inpainting_mask_invert,
                                                  img2img_batch_input_dir,
                                                  img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                                  override_settings_texts,
                                                  *sam_args)
                    for res_img in res[0]:
                        if getattr(res_img, 'already_saved_as', False):
                            if self.predict_image(res_img.already_saved_as):
                                fuck_img_count += 1
                                if fuck_img_count > 10:
                                    return {'success': False, 'result': "fatal error"}
                            else:
                                ok_img_count += 1
                                ok_res.append(res_img)

                    else:
                        img_urls = []
                        dir_path = CONFIG['storage_dirpath']['user_dir']
                        for ok_img in ok_res:
                            img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{''.join([random.choice(string.ascii_letters) for c in range(6)])}.png"
                            img_fp = f"{'localhost:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}"

                            # extra upscaler
                            scales = 1
                            gfpgan_enable = 0
                            codeformer_enable = 1
                            args = (0, scales, None, None, True, 'None', 'None', 0, gfpgan_enable, codeformer_enable, 0)
                            self.devices.torch_gc()
                            pp = self.scripts_postprocessing.PostprocessedImage(ok_img.convert("RGB"))
                            self.scripts.scripts_postproc.run(pp, args)

                            pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=100, lossless=True)
                            img_urls.append(img_fp)
                return {'success': True, 'result': img_urls}

            else:
                params = ujson.loads(kwargs['params'][0])
                # _input_image = base64_to_pil(params['input_image'])
                _input_image = Image.open(io.BytesIO(kwargs['input_image'][0][1]))
                _output_width = int(params['output_width'])
                _output_height = int(params['output_height'])

                _input_image_width, _input_image_height = _input_image.size
                _output_ratio = _output_width / _output_height
                _input_ratio = _input_image_width / _input_image_height

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
                    sd_positive_prompt = '(best quality:1.2),(high quality:1.2),masterpiece,high details,(Realism:1.4), vivid color, (realistic, photo-realistic:1.3), masterpiece'
                    sd_negative_prompt = '(extra clothes:1.5),(clothes:1.5),(NSFW:1.3),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)),(wear a mask),(cover face), duplicate, morbid, mutilated, mutated hands, poorly drawn face,skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, bad feet, multiple people, blurry, poorly drawn hands, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, too many fingers, gross proportions, abdominal stretch, fused fingers, bad body, ng_deepnegative_v1_75t, bad-picture-chill-75v, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8), aged up, old fingers, long neck, cross-eyed, polar lowres, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong'
                    prompt_styles = None
                    init_img = _input_image
                    sketch = None
                    init_img_with_mask = None
                    inpaint_color_sketch = None
                    inpaint_color_sketch_orig = None
                    init_img_inpaint = None
                    init_mask_inpaint = None
                    mask_blur = 4
                    mask_alpha = 0
                    inpainting_fill = 1
                    restore_faces = False
                    tiling = False
                    n_iter = 1
                    batch_size = 1
                    cfg_scale = 7
                    image_cfg_scale = 1.5
                    denoising_strength = 0.8
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
                    cnet_idx = 1
                    controlnet_args_unit1 = self.scripts.scripts_img2img.alwayson_scripts[
                        cnet_idx].get_default_ui_unit()
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
                                 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4,
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
                                '', True,
                                50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                                '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>',
                                128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0,
                                ['left', 'right', 'up', 'down'], False,
                                False, 'positive', 'comma', 0, False, False, '',
                                '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                                64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None,
                                False, None,
                                None, False, None, None, False, 50
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

                self.devices.torch_gc()
                # cnet_res[0][0].save(f'tmp/cnet_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                #                   format='PNG')
                # extra upscaler
                cnet_res_img = _input_image if _output_ratio == _input_ratio else cnet_res[0][0]
                scales = _output_width / padding_width
                #
                gfpgan_enable = 0
                codeformer_enable = 1
                args = (0, scales, None, None, True, 'ESRGAN_4x', 'None', 0, gfpgan_enable, codeformer_enable, 0)
                assert cnet_res_img, 'image not selected'
                self.devices.torch_gc()
                pp = self.scripts_postprocessing.PostprocessedImage(cnet_res_img.convert("RGB"))
                self.scripts.scripts_postproc.run(pp, args)

                self.devices.torch_gc()

                # return {'success': True, 'result': pil_to_base64(pp.image)}
                dir_path = CONFIG['storage_dirpath']['user_dir']
                img_fn = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{''.join([random.choice(string.ascii_letters) for c in range(6)])}.jpeg"
                img_fp = f"{'localhost:'+str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}"
                # pp.image.save(os.path.join(dir_path, img_fn), format="png", quality=100)
                pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=100, lossless=True)

                return {'success': True, 'result': [img_fp]}

        except Exception:
            print('errrrr!!!!!!!!!!!!!!')
            self.logging(
                f"[ocr predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        return {'success': False, 'result': "fatal error"}


if __name__ == '__main__':
    op = OperatorSD()
    op()
