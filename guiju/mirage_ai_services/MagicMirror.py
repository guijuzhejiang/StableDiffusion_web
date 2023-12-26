# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import datetime
import glob
import os
import random
import string
import urllib.parse

import numpy as np
import ujson
from PIL import Image, ImageDraw

from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion
from utils.global_vars import CONFIG


class MagicMirror(object):
    operator = None
    sd_model_name = 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting'

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        if self.operator.shared.sd_model.sd_checkpoint_info.model_name != self.sd_model_name:
            self.operator.shared.change_sd_model(self.sd_model_name)
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        _input_image = kwargs['input_image']
        pic_name = kwargs['pic_name']

        origin_image_path = f'tmp/{self.__class__.__name__}_origin_{pic_name}_save.png'
        _input_image.save(origin_image_path, format='PNG')
        _input_image_width, _input_image_height = _input_image.size

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

        person_boxes, _ = self.operator.dino.dino_predict_internal(_input_image, self.operator.dino_model_name, 'person', 0.3)
        if len(person_boxes) == 0:
            return {'success': False, 'result': 'backend.magic-mirror.error.no-body'}

        if self.operator.update_progress(40):
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
            # result_images = result_images[0]
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
                        if self.operator.update_progress((batch_idx + 1) * (proceed_idx + 1) * (
                                80 // (batch_size * len(task_list)))):
                            return {'success': True}
            else:
                if self.operator.update_progress(80):
                    return {'success': True}

            # storage img
            res = []
            for res_idx, res_img in enumerate(result_images):
                img_fn = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                if 'face_expression' in task_list or 'age' in task_list or 'gender' in task_list:
                    # extra upscaler
                    scales = 1
                    gfpgan_weight = 0
                    codeformer_visibility = 0.5
                    args = (
                        0, scales, None, None, True, 'None', 'None', 0, gfpgan_weight, codeformer_visibility, 0)
                    pp = self.operator.scripts_postprocessing.PostprocessedImage(res_img.convert("RGB"))
                    self.operator.scripts.scripts_postproc.run(pp, args)
                    self.operator.devices.torch_gc()
                    res.append(pp.image)
                else:
                    res.append(res_img)

            # celery_task.update_state(state='PROGRESS', meta={'progress': 95})
            if self.operator.update_progress(99):
                return {'success': True}

            return res

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
        controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[self.operator.cnet_idx].get_default_ui_unit()
        controlnet_args_unit1.enabled = False
        controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit2.enabled = False
        controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit3.enabled = False

        result_images = []
        pic_name = ''.join([random.choice(string.ascii_letters) for c in range(6)])

        if _task_type == 'gender':
            # segment
            sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "person", 0.4, _init_img)

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
            sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "person", 0.45, _init_img)
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
            # sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "face.glasses",
            #                                                 0.3, _init_img)
            sam_result = self.operator.facer(_init_img, keep='face')
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
            sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "face.glasses",
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
            # sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "hair",
            #                                                 0.36, _init_img)
            sam_result = self.operator.facer(_init_img, keep='hair')
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
            sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "breasts.arms.legs.abdomen", 0.31,
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
        res = self.operator.img2img.img2img(task_id,
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

        self.operator.devices.torch_gc()

        return res[0][:_batch_size]
