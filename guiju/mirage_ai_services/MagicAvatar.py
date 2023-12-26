# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import datetime
import os
import random
import string

import numpy as np
import ujson
from PIL import Image, ImageDraw

from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion

female_avatar_reference_dict = {
    1: {'label': 'Q版',
        'prompt': 'chibi,(3d:1.3)',
        },
    2: {'label': '漫画',
        'prompt': '',
        },
    3: {'label': '手办',
        'prompt': '(faux figurine:1.3),3d',
        },
    4: {'label': '水彩',
        'prompt': '(watercolor \(medium\):1.3)',
        },
    5: {'label': '油画',
        'prompt': '(oil painting \(medium\):1.3)',
        },
    6: {'label': '素描',
        'prompt': '(sketch,monochrome,greyscale:1.3)',
        },
    7: {'label': '蜡笔',
        'prompt': '(crayon \(medium\):1.3)',
        },
    8: {'label': '纸偶',
        'prompt': '(paper art:1.5),3d',
        },
    0: {'label': '赛博朋克',
        'prompt': '(surreal:1.5),(cyberpunk:1.5),(mecha:1.5)',
        },
}

male_avatar_reference_dict = {
    0: {'label': '赛博朋克',
        'prompt': '(surreal:1.5),(cyberpunk:1.5),(mecha:1.5)',
        },
    1: {'label': 'Q版',
        'prompt': '<lora:Guofeng_mengwan_boy:1>,chibi,1boy',
        },
    2: {'label': '亚克力',
        'prompt': '(acrylic paint \(medium\):1.3)',
        },
    3: {'label': '手办',
        'prompt': '(faux figurine:1.3),3d',
        },
    4: {'label': '泥塑',
        'prompt': '',
        },
    5: {'label': '素描',
        'prompt': '(sketch,monochrome,greyscale:1.3)',
        },
    6: {'label': '纸偶',
        'prompt': '(paper art:1.5),3d',
        },
    7: {'label': '彩墨',
        'prompt': '<lora:Chinese_Ink_Painting_style:0.6>,1boy,greyscale,ink painting,Chinese martial arts style',
        },
    8: {'label': '水彩',
        'prompt': '<lora:Watercolor_Painting_by_vizsumit:0.6>,watercolor painting',
        },
}

lora_avatar_dict = {
    '素描': {
        0: '<lora:penink:0.4>,(penink,monochrome,greyscale:1.3)',
        1: '<lora:Greyscale-000012:0.7>,(sketch,monochrome,greyscale:1.5)',
        2: '<lora:inoitohV2-000012:0.6>,(monochrome,greyscale:1.5)',
        3: '<lora:otoyomegatari:0.8>,halftone,drawing,manga,(monochrome,greyscale:1.5)',
        4: '<lora:1658606601595300147:0.6>,(sketch,monochrome,greyscale:1.5)',
        5: '<lora:1658601235832400793:0.5>,(sketch,monochrome,greyscale:1.5)',
        6: '<lora:style18-v2:0.7>,(sketch,monochrome,greyscale:1.5)',
        7: '(sketch,monochrome,greyscale:1.5)',
    },
    '泥塑': {
        0: '(acrylic paint \(medium\):1.5),<lora:Quartz_sand_acrylic_texture:1>,Quartz sand acrylic texture'
    },
    '赛博朋克': {
        0: '<lora:Steampunkcog:0.8>,meccog,cog,asians,(surreal:1.5),(cyberpunk:1.5),(mecha:1.5)',
        1: '',
    }
}


class MagicAvatar(object):
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
        _input_image = kwargs['input_image']
        pic_name = kwargs['pic_name']

        origin_image_path = f'tmp/{self.__class__.__name__}_origin_{pic_name}_save.png'
        _input_image.save(origin_image_path, format='PNG')
        _input_image_width, _input_image_height = _input_image.size

        _batch_size = int(params['batch_size'])
        _style = int(params['style'])
        _sim = float(params['sim'])
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
            person_boxes = self.operator.facer.detect_face(_input_image)
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
                f"tmp/avatar_origin_face_style{str(_style)}_sim{_sim}_gender{_gender}_{pic_name}_save.png")

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

        if self.operator.update_progress(10):
            return {'success': True}

        avatar_result = self.proceed_avatar(_input_image, _style, _gender, denoising_strength,
                                            _batch_size, pic_name, _txt2img)

        return avatar_result

    def proceed_avatar(self, _init_img, _selected_index, _gender, _denoising, _batch_size,
                       pic_name, _txt2img=False):
        task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

        # prompt setting
        sd_negative_prompt = f"(NSFW:1.8),EasyNegative, easynegative, ng_deepnegative_v1_75t,verybadimagenegative_v1.3, (worst quality:2), (low quality:2), (normal quality:2),bad anatomy, DeepNegative,text, error, cropped, mutation, deformed, jpeg artifacts,polar lowres, bad proportions, gross proportions"

        prompt_dict = female_avatar_reference_dict if _gender == 'female' else male_avatar_reference_dict
        _selected_style = prompt_dict[_selected_index]['label']
        reference_enbale = True if (_gender == 'female' and _selected_style != "素描") else False

        _selected_type = -1
        if reference_enbale:
            _reference_img_dir = os.path.join(reference_dir, f"avatar_reference", _gender, _selected_style)
            _selected_type = random.randint(0, len(os.listdir(_reference_img_dir)) - 1)

            _reference_img_rgb_ndarray = np.array(
                Image.open(os.path.join(_reference_img_dir, f"{str(_selected_type)}.jpeg")).convert('RGB'))
            _reference_img_mask_ndarray = np.zeros(shape=_reference_img_rgb_ndarray.shape)
            sd_positive_prompt = f"{prompt_dict[_selected_index]['prompt'] + ',' if prompt_dict[_selected_index]['prompt'] else ''}<lora:more_details:1>,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,strong contrast,huge_filesize,incredibly_absurdres,absurdres,highres,magazine cover,intense angle,dynamic angle,high saturation,poster"

        else:
            if (_gender == 'female' and _selected_style == '素描') or (_gender == 'male' and (
                    _selected_style == '泥塑' or (_selected_style == '赛博朋克'))):
                _selected_type = random.randint(0, len(lora_avatar_dict[_selected_style]) - 1)
                sd_positive_prompt = f"{lora_avatar_dict[_selected_style][_selected_type] + ','}{'1boy,' if _gender == 'male' else ''}<lora:more_details:1>,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,strong contrast,huge_filesize,incredibly_absurdres,absurdres,highres,magazine cover,intense angle,dynamic angle,high saturation,poster"
            else:
                if _gender == 'male' and _selected_style == '彩墨':
                    sd_positive_prompt = prompt_dict[_selected_index]['prompt']
                else:
                    sd_positive_prompt = f"{prompt_dict[_selected_index]['prompt'] + ',' if prompt_dict[_selected_index]['prompt'] else ''}{'1boy,' if _gender == 'male' and _selected_style != 'Q版' else ''}<lora:more_details:1>,(best quality:1.2),(high quality:1.2),high details,masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,strong contrast,huge_filesize,incredibly_absurdres,absurdres,highres,magazine cover,intense angle,dynamic angle,high saturation,poster"

        # logging
        self.operator.logging(
            f"[__call__][{datetime.datetime.now()}]:\n"
            f"[{pic_name}]:\n"
            f"{ujson.dumps({'sd_positive_prompt': sd_positive_prompt, '_selected_reference': _selected_type}, indent=4)}",
            f"logs/sd_webui.log")

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

        self.operator.update_progress(50)
        if _txt2img:
            # controlnet args
            controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
                self.operator.cnet_idx].get_default_ui_unit()
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
            res = self.operator.txt2img.txt2img(task_id,
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

            controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
                self.operator.cnet_idx].get_default_ui_unit()
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
            controlnet_args_unit1.control_mode = 'Balanced'
            controlnet_args_unit1.model = 'control_v11p_sd15_canny'
            controlnet_args_unit1.module = 'canny'
            controlnet_args_unit1.threshold_a = 100
            controlnet_args_unit1.threshold_b = 200

            # depth
            # controlnet_args_unit1.control_mode = 'My prompt is more important'
            # controlnet_args_unit1.model = 'control_v11f1p_sd15_depth'
            # controlnet_args_unit1.module = 'depth_midas'
            # controlnet_args_unit1.threshold_a = -1
            # controlnet_args_unit1.threshold_b = -1
            controlnet_args_unit2 = copy.deepcopy(controlnet_args_unit1)
            controlnet_args_unit2.enabled = reference_enbale
            if reference_enbale:
                controlnet_args_unit2.image = {
                    'image': _reference_img_rgb_ndarray,
                    'mask': _reference_img_mask_ndarray,
                }
                controlnet_args_unit2.model = 'ip-adapter-plus-face_sd15'
                # controlnet_args_unit2.module = 'reference_only' if _selected_style in ["手办", "蜡笔", "纸偶", "Q版",
                #                                                                        "赛博朋克"] else "reference_adain"
                controlnet_args_unit2.module = 'ip-adapter_clip_sd15'
                controlnet_args_unit2.weight = 0.8
                controlnet_args_unit2.processor_res = 512
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

            res = self.operator.img2img.img2img(task_id,
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

        self.operator.devices.torch_gc()

        return [x.convert('L') if _selected_style == '素描' else x.convert('RGBA') for x in res[0][:_batch_size]]
