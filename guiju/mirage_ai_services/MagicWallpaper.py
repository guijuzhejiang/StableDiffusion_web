# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import os
import random
import string

import numpy as np
from PIL import Image

from guiju.mirage_ai_services.MagicMirage import lora_mirage_dict
from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion


class MagicWallpaper(object):
    operator = None
    sd_model_name = 'dreamshaper_8'

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        if self.operator.shared.sd_model.sd_checkpoint_info.model_name != self.sd_model_name:
            self.operator.shared.change_sd_model(self.sd_model_name)
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        pic_name = kwargs['pic_name']

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

        if self.operator.update_progress(20):
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
        controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
            self.operator.cnet_idx].get_default_ui_unit()
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
        self.operator.logging(
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
        # controlnet_args_unit2.image = None

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
        if self.operator.update_progress(50):
            return {'success': True}

        task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
        sd_positive_prompt = f"{lora_mirage_dict[_selected_place]['prompt']},<lora:more_details:1>,science fiction,fantasy,incredible,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,(dramatic scene),(Epic composition:1.2),strong contrast,no humans,magazine cover,intense angle,dynamic angle,high saturation,poster"
        sd_negative_prompt = "(NSFW:1.8),(hands),(human),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(hair:1.3),bad_picturesm, EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), sketches, bad anatomy, DeepNegative, facing away, {Multiple people},text, error, cropped, blurry, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions"

        print("-------------------wallpaper logger-----------------")
        print(f"sd_positive_prompt: {sd_positive_prompt}")
        print(f"sd_negative_prompt: {sd_negative_prompt}")
        print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")
        # 生成
        txt2img_res = self.operator.txt2img.txt2img(task_id,
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
                                            *sam_args)[0][:_batch_size]

        # storage img
        res = []
        for res_idx, res_img in enumerate(txt2img_res):
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
            self.operator.devices.torch_gc()
            pp = self.operator.scripts_postprocessing.PostprocessedImage(res_img.convert("RGB"))
            self.operator.scripts.scripts_postproc.run(pp, args)

            self.operator.devices.torch_gc()

            res.append(pp.image)

        return res
