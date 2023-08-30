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
import os
from collections import OrderedDict
from PIL import Image

import cv2
import numpy as np

from lib.celery_workshop.operator import Operator
from utils.global_vars import CONFIG
from modules import extra_networks
import modules.script_callbacks


class OperatorSD(Operator):
    """ stable diffusion """
    num = 1
    cache = True
    cuda = True
    enable = True

    def __init__(self, gpu_idx=0):
        Operator.__init__(self)
        import importlib
        import json
        import os
        import sys
        from guiju.segment_anything_util.dino import dino_predict_internal
        from lib.stable_diffusion.util import initialize
        from modules import shared, ui_tempdir, sd_samplers, config_states, modelloader, extensions
        from modules import extra_networks_hypernet

        from modules.shared import cmd_opts
        import modules.scripts
        import modules.sd_models
        import guiju.segment_anything_util.sam
        from guiju.segment_anything_util.sam import init_sam_model

        os.makedirs(CONFIG['storage_dirpath']['user_dir'], exist_ok=True)

        os.environ['ACCELERATE'] = 'True'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        print("use gpu:" + str(gpu_idx))

        cmd_opts.gradio_debug = True
        cmd_opts.listen = True
        cmd_opts.debug_mode = True
        cmd_opts.enable_insecure_extension_access = True
        cmd_opts.xformers = True
        cmd_opts.disable_tls_verify = True
        cmd_opts.hide_ui_dir_config = True
        cmd_opts.disable_tome = False
        cmd_opts.lang = 'ch'
        cmd_opts.disable_adetailer = False
        initialize()
        guiju.segment_anything_util.sam.sam = init_sam_model()

        # ui_extra_networks.initialize()
        # ui_extra_networks.register_default_pages()

        # extra_networks.initialize()
        # extra_networks.register_default_extra_networks()
        # modules.script_callbacks.before_ui_callback()

        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            print("cleanup temp dir")

        sd_samplers.set_samplers()

        extensions.list_extensions()

        config_state_file = shared.opts.restore_config_state_file
        shared.opts.restore_config_state_file = ""
        shared.opts.save(shared.config_filename)

        if os.path.isfile(config_state_file):
            print(f"*** About to restore extension state from file: {config_state_file}")
            with open(config_state_file, "r", encoding="utf-8") as f:
                config_state = json.load(f)
                config_states.restore_extension_config(config_state)
        elif config_state_file:
            print(f"!!! Config state backup not found: {config_state_file}")

        modules.scripts.reload_scripts()
        print("load scripts")

        modules.script_callbacks.model_loaded_callback(shared.sd_model)
        print("model loaded callback")

        modelloader.load_upscalers()

        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        print("reload script modules")

        modules.sd_models.list_models()
        print("list SD models")

        shared.reload_hypernetworks()
        print("reload hypernetworks")

        extra_networks.initialize()
        extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
        print("initialize extra networks")

        # init sam
        modules.scripts.scripts_current = modules.scripts.scripts_img2img
        modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)

        cnet_idx = 1
        sam_idx = 2
        adetail_idx = 0
        modules.scripts.scripts_img2img.alwayson_scripts[0], \
        modules.scripts.scripts_img2img.alwayson_scripts[1], \
        modules.scripts.scripts_img2img.alwayson_scripts[2] \
            = modules.scripts.scripts_img2img.alwayson_scripts[sam_idx], \
              modules.scripts.scripts_img2img.alwayson_scripts[cnet_idx], \
              modules.scripts.scripts_img2img.alwayson_scripts[adetail_idx]

        # sam 24 args
        modules.scripts.scripts_img2img.alwayson_scripts[0].args_from = 7
        modules.scripts.scripts_img2img.alwayson_scripts[0].args_to = 31

        # controlnet 3 args
        modules.scripts.scripts_img2img.alwayson_scripts[1].args_from = 4
        modules.scripts.scripts_img2img.alwayson_scripts[1].args_to = 7

        # adetail 3 args
        modules.scripts.scripts_img2img.alwayson_scripts[2].args_from = 1
        modules.scripts.scripts_img2img.alwayson_scripts[2].args_to = 4

        # invisible detectmap
        shared.opts.control_net_no_detectmap = True

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
                                          cv2.BORDER_REPLICATE)
        padded_image = cv2.copyMakeBorder(padded_image,
                                          pt + padding if person_box[1] <= 8 else 0,
                                          pb + padding if 8 > original_height - person_box[3] else 0,
                                          pl + padding if person_box[0] <= 8 else 0,
                                          pr + padding if 8 > original_width - person_box[2] else 0,
                                          cv2.BORDER_REPLICATE,
                                          # value=edge_color
                                          )
        padded_image = cv2.cvtColor(np.array(padded_image), cv2.COLOR_BGRA2RGBA)
        return padded_image

    def get_prompt(self, _gender, _age, _viewpoint, _model_mode=0):
        sd_positive_prompts_dict = OrderedDict({
            'gender': [
                # female
                '1girl',
                # male
                '1boy',
            ],
            'age': [
                # child
                f'(child:1.3)',
                # youth
                f'(youth:1.3){"" if _gender else ", <lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>"}',
                # middlescent
                '(middlescent:1.3)',
            ],
            'common': [
                '(RAW photo, best quality)',
                '(realistic, photo-realistic:1.3)',
                'masterpiece',
                # f'(a naked {"man" if _gender else "woman"}:1.5)',
                f'an extremely delicate and {"handsome" if _gender else "beautiful"} {"male" if _gender else "female"}',
                'extremely detailed CG unity 8k wallpaper',
                # 'asian',
                'highres',
                'detailed fingers',
                'realistic fingers',
                # 'sleeves past wrist',
                'beautiful detailed nose',
                'beautiful detailed eyes',
                'detailed hand',
                'realistic hand',
                # 'detailed foot',
                'realistic body',
                '' if _gender else 'fluffy hair',
                '' if _viewpoint == 2 else 'posing for a photo, light on face, realistic face',
                '(simple background:1.3)',
                '(white background:1.3)',
                '(full body:1.3)' if _model_mode == 0 else '(full body:1.8)',
            ],
            'viewpoint': [
                # 正面
                'light smile',
                # 侧面
                'light smile, a side portrait photo of a people, (looking to the side:1.5)',
                # 反面
                '(a person with their back to the camera:1.5)'
            ]
        })

        sd_positive_prompts_dict['common'] = [x for x in sd_positive_prompts_dict['common'] if x]
        sd_positive_prompts_dict['gender'] = [sd_positive_prompts_dict['gender'][_gender]]
        sd_positive_prompts_dict['age'] = [sd_positive_prompts_dict['age'][_age]]
        sd_positive_prompts_dict['viewpoint'] = [sd_positive_prompts_dict['viewpoint'][_viewpoint]]

        if _viewpoint == 2:
            sd_positive_prompt = f'(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, 2k wallpaper,realistic body, (simple background:1.3), (white background:1.3), (from behind:1.3){", 1boy" if _gender else ""}'

        else:
            sd_positive_prompt = ', '.join([i for x in sd_positive_prompts_dict.values() for i in x])

        sd_negative_prompt = '(extra clothes:1.5),(clothes:1.5),(NSFW:1.3),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), clothing, pants, shorts, t-shirt, dress, sleeves, lowres, ((monochrome)), ((grayscale)), duplicate, morbid, mutilated, mutated hands, poorly drawn face,skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, too many fingers, long neck, cross-eyed, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8), aged up, old fingers, long neck, cross-eyed, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, bad body, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8)'

        return sd_positive_prompt, sd_negative_prompt

    def __call__(self, *args, **kwargs):
        import copy
        import datetime
        import math
        import random
        import string
        import traceback
        import ujson
        from guiju.predictor_opennsfw2 import predict_image
        from guiju.segment_anything_util.dino import dino_predict_internal
        from lib.common.common_util import logging, base64_to_pil, pil_to_base64
        from modules.shared import cmd_opts
        import modules.scripts
        import modules.img2img
        from guiju.segment_anything_util.sam import sam_predict
        from modules import devices, scripts_postprocessing, scripts

        try:
            # init lora
            extra_networks.initialize()
            extra_networks.register_default_extra_networks()
            modules.script_callbacks.before_ui_callback()
            print("operation start !!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(args)
            # print(kwargs)
            proceed_mode = kwargs['mode'][0]

            if proceed_mode == 'model':
                params = ujson.loads(kwargs['params'][0])
                _cloth_part = 0
                _batch_size = int(params['batch_size'])
                # _input_image = base64_to_pil(params['input_image'])
                _input_image = Image.open(io.BytesIO(kwargs['input_image'][0][1]))
                _gender = 0 if params['gender'] == 'female' else 1
                arge_idxs = {v: i for i, v in enumerate(['child', 'youth', 'middlescent'])}
                _age = arge_idxs[params['age']]
                viewpoint_mode_idxs = {v: i for i, v in enumerate(['front', 'side', 'back'])}
                _viewpoint_mode = viewpoint_mode_idxs[params['viewpoint_mode']]
                _model_mode = 0 if params['model_mode'] == 'normal' else 1

                output_height = 1024
                output_width = 512

                _dino_model_name = "GroundingDINO_SwinB (938MB)"
                # _sam_model_name = 'samhq_vit_h_1b3123.pth'

                # _dino_clothing_text_prompt = 'clothing . pants . shorts . dress . shirt . t-shirt . skirt . underwear . bra . swimsuits . bikini . stocking . chain . bow' if _model_mode == 1 else 'clothing . pants . shorts'
                _dino_clothing_text_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear . bra . swimsuit . bikini . stocking . chain . bow'
                _box_threshold = 0.3

                if _input_image is None:
                    return None, None
                else:
                    origin_image_path = f'tmp/origin_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
                    _input_image.save(origin_image_path, format='PNG')

                    try:
                        if predict_image(origin_image_path):
                            return {'success': False, 'result': "fatal error"}

                        _input_image_width, _input_image_height = _input_image.size

                        # real people
                        if _model_mode == 0:
                            person_boxes, _ = dino_predict_internal(_input_image, _dino_model_name, "person",
                                                                    _box_threshold)
                            person0_box = [int(x) for x in person_boxes[0]]
                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            top_ratio = 0.2
                            bottom_ratio = 0.2
                            if (person0_box[1] / person0_height) <= top_ratio:
                                person0_box[1] = 0
                            else:
                                person0_box[1] -= top_ratio*person0_height

                            if (_input_image_height-person0_box[3]) / person0_height <= bottom_ratio:
                                person0_box[3] = _input_image_height
                            else:
                                person0_box[3] += bottom_ratio*person0_height
                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            _input_image = self.configure_image(_input_image, person0_box, target_ratio=2 / 3 if (person0_width / person0_height) < 0.5 else person0_width / person0_height)

                            if cmd_opts.debug_mode:
                                cv2.imwrite(f'tmp/person_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', cv2.cvtColor(np.array(_input_image), cv2.COLOR_RGBA2BGRA))
                        # artificial model
                        else:
                            artificial_model_dino_clothing_prompt = 'clothing . pants . short . dress . shirt . t-shirt . skirt . underwear . bra . bikini'
                            person_boxes, _ = dino_predict_internal(_input_image, _dino_model_name,
                                                                    artificial_model_dino_clothing_prompt, _box_threshold)

                            # get max area clothing box
                            x_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 0 or i == 2]
                            y_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 1 or i == 3]
                            person0_box = [min(x_list), min(y_list), max(x_list), max(y_list)]

                            person0_width = person0_box[2] - person0_box[0]
                            person0_height = person0_box[3] - person0_box[1]
                            constant_bottom = 40
                            constant_top = 30
                            factor_bottom = 5
                            factor_top = 4
                            left_ratio = 0.1
                            right_ratio = 0.1
                            # top_ratio = 0.32
                            top_ratio = min(0.35, math.pow(person0_width / person0_height, factor_top) * constant_top)
                            bottom_ratio = min(0.58, math.pow(person0_width / person0_height,
                                                              factor_bottom) * constant_bottom)
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
                                                                     person0_box[
                                                                         2])) / person0_width) < right_ratio else padding_left +
                                                                                                                  person0_box[
                                                                                                                      2] + int(
                                                                     person0_width * right_ratio),
                                                                 _input_image_height + padding_bottom + padding_top if padding_bottom > 0 else padding_top +
                                                                                                                                               person0_box[
                                                                                                                                                   3] + int(
                                                                     person0_height * bottom_ratio)],
                                                                target_ratio=2 / 3 if (
                                                                                              person0_width / person0_height) < 0.5 else person0_width / person0_height)

                    except Exception:
                        print(traceback.format_exc())
                        print('preprocess img error')
                    else:
                        # limit height 768
                        check_h, check_w, _ = _input_image.shape
                        if check_h % 2 != 0:
                            check_h -= 1
                        if check_h > 768:
                            tmp_w = int(768 * check_w / check_h)
                            if tmp_w % 2 != 0:
                                tmp_w -= 1
                            _input_image = cv2.resize(_input_image, (tmp_w, 768))
                        # 压缩图像质量
                        quality = 100
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                        _, jpeg_data = cv2.imencode('.png', cv2.cvtColor(np.array(_input_image), cv2.COLOR_RGBA2BGRA),
                                                    encode_param)

                        # 将压缩后的图像转换为PIL图像
                        _input_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')
                        output_width, output_height = tuple(int(x) for x in _input_image.size)

                    if cmd_opts.debug_mode:
                        _input_image.save(f'tmp/resized_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                                          format='PNG')

                sam_result_tmp_png_fp = []

                sam_result_gallery, sam_result = sam_predict(_dino_model_name, _dino_clothing_text_prompt,
                                                             _box_threshold,
                                                             _input_image)

                pic_name = ''.join([random.choice(string.ascii_letters) for c in range(15)])
                if len(sam_result_gallery) == 0:
                    return {'success': False, 'result': '未检测到服装'}

                for idx, sam_mask_img in enumerate(sam_result_gallery):
                    cache_fp = f"tmp/{idx}_{pic_name}.png"
                    sam_mask_img.save(cache_fp)
                    sam_result_tmp_png_fp.append({'name': cache_fp})

                task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

                sd_positive_prompt, sd_negative_prompt = self.get_prompt(_gender, _age, _viewpoint_mode, _model_mode)

                prompt_styles = None
                # init_img = sam_result_gallery[2]
                init_img = _input_image
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
                denoising_strength = 0.7
                seed = -1.0
                subseed = -1.0
                subseed_strength = 0
                seed_resize_from_h = 0
                seed_resize_from_w = 0
                seed_enable_extras = False
                selected_scale_tab = 0
                height = output_height
                width = output_width
                scale_by = 1
                resize_mode = 2
                inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
                inpaint_full_res_padding = 0
                inpainting_mask_invert = 1
                img2img_batch_input_dir = ''
                img2img_batch_output_dir = ''
                img2img_batch_inpaint_mask_dir = ''
                override_settings_texts = []

                # controlnet args
                cnet_idx = 1
                controlnet_args_unit1 = modules.scripts.scripts_img2img.alwayson_scripts[cnet_idx].get_default_ui_unit()
                controlnet_args_unit1.batch_images = ''
                controlnet_args_unit1.control_mode = 'Balanced' if _model_mode == 0 else 'My prompt is more important'
                controlnet_args_unit1.enabled = _model_mode == 0
                # controlnet_args_unit1.enabled = False
                controlnet_args_unit1.guidance_end = 0.8
                controlnet_args_unit1.guidance_start = 0  # ending control step
                controlnet_args_unit1.image = None
                # controlnet_args_unit1.input_mode = batch_hijack.InputMode.SIMPLE
                controlnet_args_unit1.low_vram = False
                controlnet_args_unit1.model = 'control_v11p_sd15_normalbae'
                controlnet_args_unit1.module = 'normal_bae'
                controlnet_args_unit1.pixel_perfect = True
                controlnet_args_unit1.resize_mode = 'Crop and Resize'
                controlnet_args_unit1.processor_res = 512
                controlnet_args_unit1.threshold_a = 64
                controlnet_args_unit1.threshold_b = 64
                controlnet_args_unit1.weight = 1
                # controlnet_args_unit1.weight = 0.4
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
                adetail_enabled = not cmd_opts.disable_adetailer
                face_args = {'ad_model': 'face_yolov8m.pt',
                             'ad_prompt': f'',
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
                             'ad_prompt': '',
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
                    res = modules.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt, prompt_styles,
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
                                                  selected_scale_tab, height, width, scale_by, resize_mode,
                                                  inpaint_full_res,
                                                  inpaint_full_res_padding, inpainting_mask_invert,
                                                  img2img_batch_input_dir,
                                                  img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                                  override_settings_texts,
                                                  *sam_args)
                    for res_img in res[0]:
                        if getattr(res_img, 'already_saved_as', False):
                            if predict_image(res_img.already_saved_as):
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
                            img_fp = f"http://localhost:5004/user/image/fetch?imgpath={img_fn}"
                            ok_img.save(os.path.join(dir_path, img_fn), format="PNG")
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
                if _output_ratio != 0.5:
                    padding_height = int(
                        _input_image_width / _output_ratio) if _input_image_width <= _input_image_height else _input_image_height
                    padding_width = _input_image_width if _input_image_width <= _input_image_height else int(
                        _input_image_height * _output_ratio)

                    # img2img padding to _output_ratio
                    steps = 20
                    sampler_index = 18  # sampling method modules/sd_samplers_kdiffusion.py

                    task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
                    sd_positive_prompt = '(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece,extremely detailed CG unity 8k wallpaper'
                    sd_negative_prompt = '(extra clothes:1.5),(clothes:1.5),(NSFW:1.3),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), duplicate, morbid, mutilated, mutated hands, poorly drawn face,skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, bad feet, multiple people, blurry, poorly drawn hands, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, too many fingers, gross proportions, abdominal stretch, fused fingers, bad body, ng_deepnegative_v1_75t, bad-picture-chill-75v, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8), aged up, old fingers, long neck, cross-eyed, polar lowres, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong'
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
                    controlnet_args_unit1 = modules.scripts.scripts_img2img.alwayson_scripts[
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

                    cnet_res = modules.img2img.img2img(task_id, 0, sd_positive_prompt, sd_negative_prompt,
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
                    padding_height = _input_image_height
                    padding_width = _input_image_width

                # extra upscaler
                cnet_res_img = _input_image if _output_ratio == 0.5 else cnet_res[0][0]
                scales = _output_width / padding_width
                args = (0, scales, padding_height, padding_width, True, 'ESRGAN_4x', 'None', 0, 0, 0, 0)
                assert cnet_res_img, 'image not selected'

                devices.torch_gc()
                image_data = []
                image_names = []
                image_data.append(cnet_res_img)
                image_names.append(None)
                # outpath = shared.opts.outdir_samples or shared.opts.outdir_extras_samples

                pp = scripts_postprocessing.PostprocessedImage(cnet_res_img.convert("RGB"))

                scripts.scripts_postproc.run(pp, args)

                # if save_output:
                # pp.image.save('test.png')

                devices.torch_gc()

                # return {'success': True, 'result': pil_to_base64(pp.image)}
                dir_path = CONFIG['storage_dirpath']['user_dir']
                img_fn = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{''.join([random.choice(string.ascii_letters) for c in range(6)])}.png"
                img_fp = f"http://localhost:5004/user/image/fetch?imgpath={img_fn}"
                pp.image.save(os.path.join(dir_path, img_fn), format="PNG")
                return {'success': True, 'result': [img_fp]}

        except Exception:
            print('errrrr!!!!!!!!!!!!!!')
            logging(
                f"[ocr predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        return {'success': False, 'result': "fatal error"}


if __name__ == '__main__':
    op = OperatorSD()
    op()
