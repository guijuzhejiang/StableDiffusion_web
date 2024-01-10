# coding=utf-8
# @Time : 2023/12/15 下午3:15
# @File : magic_cert.py
import copy
import os
import random
import re
import string
import datetime

import numpy as np

from utils.global_vars import CONFIG


class MagicHires(object):
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

        pattern = re.compile(r"user_(.*?)_history")
        match = pattern.search(_input_image_path)

        if match:
            _input_image_mode = match.group(1)
        else:
            _input_image_mode = 'facer'
        # hires
        if _input_image_mode == 'avatar' or _input_image_mode == 'mirage':
            if self.operator.shared.sd_model.sd_checkpoint_info.model_name != 'dreamshaper_8':
                self.operator.shared.change_sd_model('dreamshaper_8')
        elif _input_image_mode == 'facer':
            pass
        else:
            if self.operator.shared.sd_model.sd_checkpoint_info.model_name != 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting':
                self.operator.shared.change_sd_model('chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting')

        # _input_image = base64_to_pil(params['input_image'])
        _output_width = int(params['output_width'])
        _output_height = int(params['output_height'])

        _output_ratio = _output_width / _output_height
        _input_ratio = _input_image_width / _input_image_height

        # celery_task.update_state(state='PROGRESS', meta={'progress': 10})
        if self.operator.update_progress(10):
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
            controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
                self.operator.cnet_idx].get_default_ui_unit()
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

            cnet_res = self.operator.img2img.img2img(task_id, 0, sd_positive_prompt, sd_negative_prompt,
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
        if self.operator.update_progress(50):
            return {'success': True}
        self.operator.devices.torch_gc()
        # cnet_res[0][0].save(f'tmp/cnet_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
        #                   format='PNG')
        # extra upscaler
        cnet_res_img = _input_image if _output_ratio == _input_ratio else cnet_res[0][0]
        scales = _output_width / padding_width

        # celery_task.update_state(state='PROGRESS', meta={'progress': 70})
        if self.operator.update_progress(70):
            return {'success': True}

        gfpgan_weight = 0
        codeformer_visibility = 1 if _input_image_mode == 'model' else 0
        args = (0, scales, None, None, True, 'ESRGAN 4x', 'None', 0, gfpgan_weight, codeformer_visibility,
                0 if _input_image_mode == 'model' else 1)
        assert cnet_res_img, 'image not selected'
        self.operator.devices.torch_gc()
        pp = self.operator.scripts_postprocessing.PostprocessedImage(cnet_res_img.convert("RGB"))
        self.operator.scripts.scripts_postproc.run(pp, args)

        self.operator.devices.torch_gc()

        # celery_task.update_state(state='PROGRESS', meta={'progress': 80})
        if self.operator.update_progress(80):
            return {'success': True}

        dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], 'hires')

        os.makedirs(dir_path, exist_ok=True)

        img_fn = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{''.join([random.choice(string.ascii_letters) for c in range(6)])}.jpeg"
        img_fp = f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}"
        # pp.image.save(os.path.join(dir_path, img_fn), format="png", quality=100)
        pp.image.save(os.path.join(dir_path, img_fn), format="jpeg", quality=100, lossless=True)
        # celery_task.update_state(state='PROGRESS', meta={'progress': 90})
        if self.operator.update_progress(90):
            return {'success': True}
        return {'success': True, 'result': [img_fp]}
