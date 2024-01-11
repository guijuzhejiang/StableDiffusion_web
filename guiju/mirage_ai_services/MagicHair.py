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

# 头发 fluffy hair,lush hair,
lora_haircut_common_dict = {
    'positive_prompt': '(simple background:1.3),(Realism:1.4),masterpiece,raw photo,realistic,(best quality:1.2),(high quality:1.2),<lora:more_details:1>,black shirt',
    'negative_prompt': '(jewelry:1.5),(earrings:1.5),(stud earrings:1.5),cat_ears,(NSFW:1.8),(hands:1.5),(feet:1.3),(shoes:1.3),(mask:1.3),(glove:1.3),(fingers:1.3),(legs),(toes:1.3),(digits:1.3),bad_picturesm,EasyNegative,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3,(worst quality:2),(low quality:2),(normal quality:2),((monochrome)),((grayscale)),sketches,bad anatomy,DeepNegative,{Multiple people},text,error,cropped,blurry,mutation,deformed,jpeg artifacts,polar lowres,bad proportions,gross proportions,humans'
}

lora_haircut_male_dict = {
    9: {'label': '短脏辫',
        'prompt': '<lora:short_dreads_hairstyle:0.5>,(short_dreads_hairstyle:1.3)',
        },
    10: {'label': '长脏辫',
         'prompt': '<lora:dreads_hairstyle:0.5>,(dreads_hairstyle:1.3)',
         },
    8: {'label': '冲天发',
        'prompt': '<lora:hi_top_fade_hairstyle:0.5>,(hi_top_fade_hairstyle:1.3)',
        },
    3: {'label': '爆炸头',
        'prompt': '<lora:afro_hairstyle:0.5>,(afro_hairstyle:1.3)',
        },
    4: {'label': '寸头',
        'prompt': '<lora:buzzcut_hairstyle:0.5>,(buzzcut_hairstyle:1.3)',
        },
    5: {'label': '偏分',
        'prompt': '<lora:a_line_hairstyle:0.5>,(a_line_haircut:1.3)',
        },
    6: {'label': '寸鬓',
        'prompt': '<lora:half_buzzcut_hairstyle:0.5>,(half_buzzcut_hairstyle:1.3)',
        },
    7: {'label': '中发',
        'prompt': '<lora:wolf_cut:0.8>,wolf_cut',
        },
    1: {'label': '小卷',
        'prompt': '<lora:wavy_centerparted:1>,wavy_centerparted',
        },
    0: {'label': '大卷',
        'prompt': '<lora:wavy_bangs:1>,wavy_bangs',
        },
    2: {'label': '微卷',
        'prompt': '<lora:short_hair:1>,short hair',
        },
}

lora_haircut_female_dict = {
    22: {'label': '短脏辫',
         'prompt': '<lora:short_dreads_hairstyle:0.5>,(short_dreads_hairstyle:1.3)',
         },
    23: {'label': '长脏辫',
         'prompt': '<lora:dreads_hairstyle:0.5>,(dreads_hairstyle:1.3)',
         },
    14: {'label': '中分',
         'prompt': '<lora:middle_parting_hairstyle:0.5>,(middle_parting_hairstyle:1.3)',
         },
    3: {'label': '上抓头',
        'prompt': '<lora:hi_top_fade_hairstyle:0.5>,(hi_top_fade_hairstyle:1.3)',
        },
    4: {'label': '哪吒头',
        'prompt': '<lora:side_buns_hairstyle:0.5>,(side_buns_hairstyle:1.3)',
        },
    5: {'label': '爆炸头',
        'prompt': '<lora:afro_hairstyle:0.5>,(afro_hairstyle:1.3)',
        },
    6: {'label': '长辫子',
        'prompt': '<lora:long_braid_hairstyle-10:0.5>,(long_braid_hairstyle:1.3)',
        },
    7: {'label': '丸子头',
        'prompt': '<lora:space_buns_hairstyle:1>,(space_buns_hairstyle:1.3)',
        },
    21: {'label': '黑人辫',
         'prompt': '<lora:knotless_braid_hairstyle:0.5>,(knotless_braid_hairstyle:1.3)',
         },
    9: {'label': '新娘发型',
        'prompt': '<lora:bridal_hairstyle-10:0.5>,(bridal_hairstyle:1.3)',
        },
    10: {'label': '短马尾',
         'prompt': '<lora:short_pigtail_hairstyle05:0.5>,(short_pigtail_hairstyle:1.3)',
         },
    11: {'label': '盘发',
         'prompt': '<lora:updo_hairstyle:0.5>,(updo_hairstyle:1.3)',
         },
    12: {'label': '短发',
         'prompt': '<lora:pixie_hairstyle-05:0.5>,(pixie_hairstyle:1.3)',
         },
    13: {'label': '双马尾',
         'prompt': '<lora:pigtail_hairstyle:0.5>,(pigtail_hairstyle:1.3)',
         },
    0: {'label': '大卷',
        'prompt': '<lora:curls_hairstyle-10:0.5>,(curls_hairstyle:1.3)',
        },
    15: {'label': '寸头',
         'prompt': '<lora:buzzcut_hairstyle:0.5>,(buzzcut_hairstyle:1.3)',
         },
    16: {'label': '单马尾',
         'prompt': '<lora:ponytail_weave_hairstyle:0.5>,(ponytail_weave_hairstyle:1.3)',
         },
    # 17: {'label': '及肩短发',
    #     'prompt': '<lora:egyptian_bob_hairstyle:0.5>,(egyptian_bob_hairstyle:1.3)',
    #     },
    17: {'label': '偏分',
         'prompt': '<lora:a_line_hairstyle:0.5>,(a_line_haircut:1.3)',
         },
    18: {'label': '超长直发',
         'prompt': '<lora:very_long_hair-10:0.5>,(very_long_hair:1.3)',
         },
    19: {'label': '寸鬓',
         'prompt': '<lora:half_buzzcut_hairstyle:0.5>,(half_buzzcut_hairstyle:1.3)',
         },
    20: {'label': '长马尾',
         'prompt': '<lora:long_ponytail_hairstyle:0.5>,(long_ponytail_hairstyle:1.3)',
         },
    8: {'label': '包耳短发',
        'prompt': '<lora:SBobHaircut:0.8>,short hair',
        },
    2: {'label': '微卷短发',
        'prompt': '<lora:wolf_cut:0.8>,wolf_cut',
        },
    1: {'label': '大卷短发',
        'prompt': '<lora:wavy_bangs:1>,wavy_bangs',
        },
}

lora_hair_color_dict = {
    0: {'label': '黑色',
        'prompt': '(black hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
        },
    1: {'label': '黄色',
        'prompt': '(yellow hair:1.3)',
        },
    2: {'label': '棕色',
        'prompt': '(brown hair:1.3)',
        },
    3: {'label': '紫色',
        'prompt': '(purple hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
        },
    4: {'label': '灰色',
        'prompt': '(gray hair:1.3)',
        },
    5: {'label': '红色',
        'prompt': '(red hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
        },
    6: {'label': '深红色',
        'prompt': '(dark red hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
        },
    7: {'label': '蓝色',
        'prompt': 'blue pubic hair,(blue hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
        },
    8: {'label': '深蓝色',
        'prompt': '(dark blue hair:1.3)',
        },
    9: {'label': '绿色',
        'prompt': 'green pubic hair,(green hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
        },
    10: {'label': '橙色',
         'prompt': 'orange pubic hair,(orange hair:1.4),ultra high resolution,<lora:dyed_pubes-10:1>',
         },
    11: {'label': '栗色',
         'prompt': '(maroon  hair:1.3), ultra high resolution,<lora:dyed_pubes-10:1>',
         },
    12: {'label': '金色',
         'prompt': '(golden hair:1.3)',
         },
    13: {'label': '银色',
         'prompt': '(silver hair:1.3)',
         },
    14: {'label': '亚麻色',
         'prompt': '(linen hair:1.3)',
         },
    15: {'label': '红棕色',
         'prompt': '(reddish brown  hair:1.3), ultra high resolution,<lora:dyed_pubes-10:1>',
         },
    16: {'label': '黑紫色',
         'prompt': '(black-purple hair:1.3), ultra high resolution,<lora:dyed_pubes-10:1>',
         },
    17: {'label': '橙红色',
         'prompt': '(orange red hair:1.3), ultra high resolution,<lora:dyed_pubes-10:1>',
         },
    18: {'label': '浅粉色',
         'prompt': '(light pink hair:1.3)',
         },
    19: {'label': '粉红色',
         'prompt': 'pink pubic hair,(pink hair:1.3),ultra high resolution,<lora:dyed_pubes-10:1>',
         },
}


class MagicHair(object):
    operator = None
    sd_model_name = 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting'

    def __init__(self, operator):
        self.operator = operator

    def apply_mask(self, original_image, mask_image):
        transparent_im = Image.new('RGBA', original_image.size, (255, 255, 255))
        transparent_im.paste(original_image, mask=mask_image)

        return transparent_im

    def expand_canvas(self, original_image, top_px, bottom_px, left_px, right_px):
        # 获取原始图像的尺寸
        width, height = original_image.size

        # 计算新的画布尺寸
        new_width = width + left_px + right_px
        new_height = height + top_px + bottom_px

        # 创建一个新的画布
        new_canvas = Image.new('RGB', (new_width, new_height), (255, 255, 255))

        # 将原始图像粘贴到新的画布中央
        new_canvas.paste(original_image, (left_px, top_px))

        return new_canvas

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

            if _haircut_enable:
                person_boxes = self.operator.facer.detect_face(_input_image)

            else:
                person_boxes = self.operator.facer.detect_head(_input_image)

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

                person_box = person_boxes[0]
                person_width = person_box[2] - person_box[0]
                person_height = person_box[3] - person_box[1]

                new_person_box = [0, 0, 0, 0]

                # crop
                pre_padding = 0.1
                if _haircut_enable:
                    new_person_box[0] = person_box[0] - int(person_width * pre_padding)
                    new_person_box[1] = person_box[1] - int(person_height * pre_padding)
                    new_person_box[2] = person_box[2] + int(person_width * pre_padding)
                    new_person_box[3] = person_box[3] + int(person_height * pre_padding)
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
                    new_person_box[0] = person_box[0] - int(person_width * pre_padding)
                    new_person_box[1] = person_box[1] - int(person_height * pre_padding)
                    new_person_box[2] = person_box[2] + int(person_width * pre_padding)
                    new_person_box[3] = person_box[3] + int(person_height * pre_padding)
                    if new_person_box[0] < 0:
                        new_person_box[0] = 0
                    if new_person_box[1] < 0:
                        new_person_box[1] = 0
                    if new_person_box[2] > _input_image_width - 1:
                        new_person_box[2] = _input_image_width - 1
                    if new_person_box[3] > _input_image_height - 1:
                        new_person_box[3] = _input_image_height - 1

                    _input_image = _input_image.crop(new_person_box)


            if self.operator.update_progress(10):
                return {'success': True}

            hair_result = []

            # limit 512
            _input_image_width, _input_image_height = _input_image.size
            min_edge = min(_input_image_width, _input_image_height)
            min_index = [_input_image_width, _input_image_height].index(min_edge)
            if min_index == 0:
                _input_image = _input_image.resize(
                    (512, int(_input_image_height / _input_image_width * 512)))
            else:
                _input_image = _input_image.resize(
                    (int(_input_image_width / _input_image_height * 512), 512))

            _input_image_width, _input_image_height = _input_image.size

            # haircut
            if _haircut_enable:
                _input_image = self.expand_canvas(_input_image,
                                                  int(_input_image_height*0.4),
                                                  int(_input_image_height*0.5),
                                                  int(_input_image_width*0.4),
                                                  int(_input_image_width*0.4))
                _input_image_width, _input_image_height = _input_image.size

                if _input_image_width != _input_image_height:
                    if _input_image_width > _input_image_height:
                        _input_image = self.expand_canvas(_input_image,
                                                          int((_input_image_width - _input_image_height) / 2),
                                                          _input_image_width - _input_image_height - int(
                                                              (_input_image_width - _input_image_height) / 2),
                                                          0, 0
                                                          )
                    else:
                        _input_image = self.expand_canvas(_input_image,
                                                          0,0,
                                                          int((_input_image_height - _input_image_width) / 2),
                                                          _input_image_height - _input_image_width - int(
                                                              (_input_image_height - _input_image_width) / 2))


                hair_result = self.proceed_hair(_haircut_style, 'haircut', _batch_size, _input_image, pic_name,
                                                return_list=True, gender=_gender)
                if isinstance(hair_result, dict):
                    return hair_result

            #
            cache_fp = f"tmp/hair_resized_{pic_name}_save.png"
            _input_image.save(cache_fp)

            if self.operator.update_progress(50):
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

            return hair_result

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
        controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
            self.operator.cnet_idx].get_default_ui_unit()
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
        sampler_index = 15 if _task_type == 'haircut' else 16
        inpaint_full_res = 0 if _task_type == 'haircut' else 1  # choices=["Whole picture", "Only masked"]
        inpainting_fill = 2  # masked content original
        denoising_strength = 0.85 if _task_type == 'haircut' else 0.8
        steps = 20

        if _task_type == 'haircut':
            # 切割face.glasses
            sam_result = self.operator.facer(_init_img, keep='face')

        else:
            # 切割hair
            # sam_result, person_boxes = self.operator.sam_h.sam_predict(self.operator.dino_model_name, 'hair',
            #                                                   0.4,
            #                                                   _init_img.convert('RGBA'))
            sam_result = self.operator.facer(_init_img, keep='hair')
        # _init_img = sam_result[2].convert('RGBA')

        sam_result_tmp_png_fp = []
        if sam_result is not None:
            # sam_image = self.apply_mask(_init_img, sam_result)
            for idx in range(3):
                cache_fp = f"tmp/hair_{_task_type}_{idx}_{uid_name}_{_pic_name}{'_save' if idx == 1 else ''}.png"
                if idx == 1:
                    sam_result.save(cache_fp, format='PNG')
                else:
                    sam_result.save(cache_fp, format='PNG')
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
            sd_positive_prompt = ','.join(
                ['hair',
                 lora_haircut_male_dict[_selected_index]['prompt'] if gender == 'male' else
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
            controlnet_args_unit1.control_mode = 'My prompt is more important'
            controlnet_args_unit1.guidance_end = 1
            controlnet_args_unit1.guidance_start = 0  # ending control step
            controlnet_args_unit1.image = None
            controlnet_args_unit1.low_vram = False
            controlnet_args_unit1.model = 'control_v11p_sd15_canny'
            controlnet_args_unit1.module = 'canny'
            controlnet_args_unit1.pixel_perfect = True
            controlnet_args_unit1.resize_mode = 'Resize and Fill'
            controlnet_args_unit1.processor_res = 512
            controlnet_args_unit1.threshold_a = 100
            controlnet_args_unit1.threshold_b = 200
            controlnet_args_unit1.weight = 1

            controlnet_args_unit2.enabled = True
            controlnet_args_unit2.threshold_a = 0.5
            controlnet_args_unit2.threshold_b = -1
            controlnet_args_unit2.model = 'None'
            controlnet_args_unit2.module = 'reference_adain+attn'
            controlnet_args_unit2.pixel_perfect = True
            controlnet_args_unit2.weight = 1
            controlnet_args_unit2.resize_mode = 'Crop and Resize'
            controlnet_args_unit2.image = {
                'image': np.array(_init_img),
                'mask': np.zeros(shape=np.array(_init_img).shape),
            }

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

        if return_list:
            return [x.convert('RGBA') for x in res[0][:_batch_size]]
        else:
            return res[0][0].convert('RGBA')
