# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import random
import string

from guiju.mirage_ai_services.MagicText2Image import common_prompts, text2image_style_prompts, sd_negative_prompt
from modules.sd_samplers_kdiffusion import samplers_k_diffusion


class MagicText2Image(object):
    operator = None
    sd_model_name = 'juggernautXL_v9Rundiffusionphoto2'

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
        _input_image = kwargs['input_image']

        origin_image_path = f'tmp/{self.__class__.__name__}_origin_{pic_name}_save.png'
        _input_image.save(origin_image_path, format='PNG')

        _batch_size = int(params['batchSize'])
        _output_width = int(params['width'])
        _output_height = int(params['height'])
        _style = int(params['style'])
        _prompt = params['prompt']
        _sim = float(params['sim'])

        _selected_aspect = _output_width / _output_height
        _input_image_width, _input_image_height = _input_image.size

        denoising_strength = (1 - _sim)

        if self.operator.update_progress(20):
            return {'success': True}

        # img2img generate bg
        prompt_styles = None
        steps = 20
        init_img = _input_image
        sampler_index = 16  # sampling method modules/sd_samplers_kdiffusion.py
        restore_faces = False
        tiling = False
        n_iter = 1
        cfg_scale = 6
        seed = -1.0
        subseed = -1.0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        sketch = None
        init_img_with_mask = None
        inpaint_color_sketch = None
        inpaint_color_sketch_orig = None
        init_img_inpaint = None
        init_mask_inpaint = None
        mask_blur = 0
        mask_alpha = 0
        inpainting_fill = 1
        image_cfg_scale = 1.5
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

        # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
        if self.operator.update_progress(50):
            return {'success': True}

        task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
        _common_prompt = ','.join([x for x in common_prompts if x not in text2image_style_prompts[_style]['disallow']])
        sd_positive_prompt = ','.join([text2image_style_prompts[_style]['prompt'], _prompt, _common_prompt])

        print("-------------------image2image logger-----------------")
        print(f"sd_positive_prompt: {sd_positive_prompt}")
        print(f"sd_negative_prompt: {sd_negative_prompt}")
        print(f"Sampling method: {samplers_k_diffusion[sampler_index]}")
        self.operator.logging(
            f"[{pic_name}]:\nsd_positive_prompt: {sd_positive_prompt}\nsd_negative_prompt: {sd_negative_prompt}\nSampling method: {samplers_k_diffusion[sampler_index]}",
            f"logs/sd_webui.log")

        # controlnet args
        controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
            self.operator.cnet_idx].get_default_ui_unit()
        # depth
        controlnet_args_unit1.enabled = False
        controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
        controlnet_args_unit3 = copy.deepcopy(controlnet_args_unit1)
        sam_args = [0,
                    False, {}, {},  # adetail args
                    controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                    # sam
                    False,  # inpaint_upload_enable
                    False, 0, None,
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

        # 生成
        i2i_res = self.operator.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt,
                                            prompt_styles,
                                            init_img,
                                            sketch,
                                            init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                            init_img_inpaint, init_mask_inpaint,
                                            steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                            restore_faces,
                                            tiling,
                                            n_iter, _batch_size, cfg_scale, image_cfg_scale,
                                            denoising_strength,
                                            seed,
                                            subseed,
                                            subseed_strength, seed_resize_from_h, seed_resize_from_w,
                                            seed_enable_extras,
                                            selected_scale_tab, _output_height, _output_width, scale_by,
                                            resize_mode,
                                            inpaint_full_res,
                                            inpaint_full_res_padding, inpainting_mask_invert,
                                            img2img_batch_input_dir,
                                            img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                            override_settings_texts,
                                            *sam_args)[0][:_batch_size]

        return [x.convert('RGB') for x in i2i_res[0][:_batch_size]]
