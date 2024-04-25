# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import datetime
import glob
import io
import math
import os
import random
import string
import traceback
import urllib.parse
from collections import OrderedDict

import cv2
import numpy as np
import ujson
from PIL import Image, ImageDraw

from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion
from utils.global_vars import CONFIG

lora_model_common_dict = [
    {'lora_name': 'polyhedron_new_skin_v1.1', 'weight': 0.1, 'label': '赋予真实皮肤，带褶皱'},
    # {'lora_name': 'ClothingAdjuster3', 'weight': 1, 'label': '不填加额外衣服'},
    {'lora_name': 'more_details', 'weight': 0.8, 'label': '增加细节'},
    # {'lora_name': 'k Hand Mix 101_v1.0', 'weight': 0.6, 'label': '手部修复'},
    # {'lora_name': 'Xian-T手部修复lora（不用controlnet也不坏手了）_v3.0', 'weight': 1, 'label': '手部修复'},
    # {'lora_name': 'neg4all_bdsqlsz_V3.5', 'weight': 1, 'label': '手部修复'},
]

lora_bg_common_dict = [
    # {'lora_name': 'add_detail', 'weight': 1, 'label': '增加细节'},
    {'lora_name': 'more_details', 'weight': 0.8, 'label': '增加细节'},
    {'lora_name': 'ClothingAdjuster3', 'weight': 1, 'label': '不填加额外衣服'},
]

lora_gender_dict = [
    # 女美化眼睛
    "<lora:detailed_eye-10:0.1>,1girl",
    # 男美化眼睛
    '<lora:polyhedron_men_eyes:0.1>,1man,1boy'
]

# gender 男1女0
lora_model_dict = {
    0: {'lora_name': 'shojovibe_v11',
        'weight': 0.4,
        'prompt': '',
        'gender': 0,
        'label': '亚洲少女'
        },

    1: {'lora_name': 'abd',
        'weight': 1,
        'prompt': 'beautiful abd_woman,abd_body,perfect abd_face',
        'gender': 0,
        'label': '非洲'
        },

    2: {'lora_name': 'dollface',
        'weight': 1,
        'prompt': 'irish',
        'gender': 0,
        'label': '爱尔兰人',
        },

    3: {'lora_name': 'dollface',
        'weight': 1,
        'prompt': 'spanish',
        'gender': 0,
        'label': '西班牙人'
        },

    4: {'lora_name': 'edgAustralianDoll',
        'weight': 0.7,
        'prompt': 'beautiful blonde edgAus_woman',
        'gender': 0,
        'label': '澳大利亚',
        },

    5: {'lora_name': 'edgBulgarian_Doll_Likeness',
        'weight': 1,
        'prompt': 'edgBulgr_woman,edgBulgr_face,edgBulgr_body',
        'gender': 0,
        'label': '保加利亚女性',
        },

    6: {'lora_name': 'edgEgyptian_Doll',
        'weight': 0.7,
        'prompt': 'beautiful dark haired edgEgyptian_woman,perfect edgEgyptian_face,perfect edgEgyptian_body',
        'gender': 0,
        'label': '埃及',
        },

    7: {'lora_name': 'edgIndonesianDollLikeness',
        'weight': 1,
        'prompt': 'edgIndo_woman,edgIndo_face,edgIndo_body',
        'gender': 0,
        'label': '印尼',
        },

    8: {'lora_name': 'edg_LatinaDollLikeness',
        'weight': 0.8,
        'prompt': 'beautiful Lnd_woman,perfect Lnd_face,perfect Lnd_body',
        'gender': 0,
        'label': '拉丁美洲',
        },

    9: {'lora_name': 'edgPersian',
        'weight': 1,
        'prompt': 'beautiful edgPersian_woman,perfect edgPersian_face,perfect edgPersian_body',
        'gender': 0,
        'label': '波斯',
        },

    10: {'lora_name': 'edgSwedishDoll',
         'weight': 0.7,
         'prompt': 'beautiful edgSwedish_woman,perfect edgSwedish_face,perfect edgSwedish_body',
         'gender': 0,
         'label': '瑞典'
         },

    11: {'lora_name': 'EnglishDollLikeness_v10',
         'weight': 1,
         'prompt': '',
         'gender': 0,
         'label': '英国'
         },

    12: {'lora_name': 'esd',
         'weight': 1,
         'prompt': 'beautiful esd_woman,perfect esd_face,perfect esd_body',
         'gender': 0,
         'label': '西班牙'
         },

    13: {'lora_name': 'frd',
         'weight': 0.8,
         'prompt': 'beautiful frd_woman,perfect frd_face,perfect frd_body',
         'gender': 0,
         'label': '法国'
         },

    14: {'lora_name': 'BLONDBOY',
         'weight': 1,
         'prompt': 'blondboy',
         'gender': 1,
         'label': '金发男'
         },

    15: {'lora_name': 'grd',
         'weight': 1,
         'prompt': 'beautiful grd_woman,perfect grd_face,perfect grd_body',
         'gender': 0,
         'label': '德国'
         },
    # delete
    16: {'lora_name': 'hld',
         'weight': 1,
         'prompt': 'beautiful hld_woman,perfect hld_face,perfect hld_body',
         'gender': 0,
         'label': '苏格兰高地'
         },
    17: {'lora_name': 'ind',
         'weight': 1,
         'prompt': 'beautiful ind_woman,perfect ind_face,perfect ind_body',
         'gender': 0,
         'label': '印度'
         },

    18: {'lora_name': 'IndonesianDollLikeness_V1',
         'weight': 0.7,
         'prompt': '',
         'gender': 0,
         'label': '印尼'
         },

    19: {'lora_name': 'ird',
         'weight': 0.8,
         'prompt': 'beautiful ird_woman,perfect ird_face,perfect ird_body',
         'gender': 0,
         'label': '爱尔兰'
         },

    20: {'lora_name': 'itd1',
         'weight': 1,
         'prompt': 'beautiful itd_woman,perfect itd_face,perfect itd_body',
         'gender': 0,
         'label': '意大利'
         },

    21: {'lora_name': 'koreanDollLikeness',
         'weight': 0.7,
         'prompt': '',
         'gender': 0,
         'label': '韩国'
         },

    22: {'lora_name': 'Korean Men Dolllikeness 1.0',
         'weight': 1,
         'prompt': '',
         'gender': 1,
         'label': '韩国'
         },

    23: {'lora_name': 'Lora-Custom-ModelLiXian',
         'weight': 0.5,
         'prompt': '',
         'gender': 1,
         'label': '亚洲'
         },

    24: {'lora_name': 'm3d',
         'weight': 1,
         'prompt': 'beautiful m3d_woman,perfect m3d_body,perfact m3d_face',
         'gender': 0,
         'label': '中东'
         },

    25: {'lora_name': 'nod',
         'weight': 0.7,
         'prompt': 'beautiful nod_woman,perfect nod_body,perfect nod_face',
         'gender': 0,
         'label': '挪威'
         },

    26: {'lora_name': 'PrettyBoy',
         'weight': 1,
         'prompt': 'pretty boy,caucasian,black,asian,indian',
         'gender': 1,
         'label': '白种人、黑人、亚洲人、印度人'
         },

    27: {'lora_name': 'rud',
         'weight': 1,
         'prompt': 'beautiful rud_woman,perfect rud_face,perfect rud_body',
         'gender': 0,
         'label': '俄罗斯'
         },

    28: {'lora_name': 'RussianDollV3',
         'weight': 0.8,
         'prompt': 'russian',
         'gender': 0,
         'label': '俄罗斯'
         },

    29: {'lora_name': 'syahasianV3',
         'weight': 1,
         'prompt': 'syahmi',
         'gender': 1,
         'label': '亚洲，东南亚'
         },

    30: {'lora_name': 'tkd',
         'weight': 0.7,
         'prompt': 'beautiful tkd_woman,perfect tkd_face,perfect tkd_body',
         'gender': 0,
         'label': '土耳其'
         },

    31: {'lora_name': 'VietnameseDollLikeness-v1.0',
         'weight': 0.7,
         'prompt': '',
         'gender': 0,
         'label': '越南'
         },
}

lora_place_dict = {
    0: {'label': '无背景',
        'prompt': '(simple background:1.3),(white background:1.3)'
        },
    1: {'label': '公路风光',
        'prompt': '(scenicroad:1.3),<lora:scenicroad:1.0>,landscape,a road with many trees on both sides,Utah,Florida,California,New England,Colorado,Arizona,Texas,Oregon,Pennsylvania,Washington,outdoor,professional nature photography,calm atmosphere'
        },
    2: {'label': '樱花绽放',
        'prompt': 'CherryBlossom_background,<lora:CherryBlossom_v1:0.6>,cherry blossoms in bloom,outdoor,professional nature photography,calm atmosphere,landscape',
        },
    3: {'label': '光晕',
        'prompt': 'glowingdust,bokeh,<lora:glowingdust:0.9>,outdoor,professional nature photography,calm atmosphere,landscape',
        },
    4: {'label': '公园',
        'prompt': 'Park_Bench_background,<lora:ParkBench_v1:0.6>,park,professional nature photography,calm atmosphere,landscape',
        },
    5: {'label': '天台',
        'prompt': '<lora:school_rooftop_v0.1:1> school rooftop,(rooftop:1.3),professional nature photography,calm atmosphere,chain-link fence,building',
        },
    6: {'label': '林间小路',
        'prompt': 'slg,(forest),path,<lora:slg_v30:1>,(path in woods:1.3),outdoor,professional nature photography,calm atmosphere',
        },
    7: {'label': '林间溪流',
        'prompt': 'slg,forest,(river:1.3),(stream),<lora:slg_v30:1>,outdoor,professional nature photography,calm atmosphere',
        },
    8: {'label': '林间瀑布',
        'prompt': 'slg,(waterfall:1.3),river,<lora:slg_v30:1>,huge waterfall,outdoor,professional nature photography,calm atmosphere',
        },
    9: {'label': '黄昏',
        'prompt': 'sunset_scenery_background,<lora:SunsetScenery_v1:0.6>,sunset,outdoor,professional nature photography,calm atmosphere,landscape',
        },
    10: {'label': '花团锦簇',
         'prompt': '<lora:乐章五部曲-林V1:1>,blue sky,outdoor,tree,nice bokeh professional nature photography,Cute landscape,calm atmosphere,peaceful theme,sen,nature,flowers',
         },
    11: {'label': '向日葵海',
         'prompt': 'sunflower_background,<lora:Sunflower_v1:0.4>,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
         },
    12: {'label': '沙滩',
         'prompt': 'beach,<lora:Taketomijima:1>,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
         },
    13: {'label': '夏威夷热',
         'prompt': 'tropical_tiki_retreat,<lora:tropical_tiki_retreat-10:1>,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme,Exotic,Hawaiian,aloha',
         },
    14: {'label': '水族馆',
         'prompt': 'aquarium,<lora:Aquarium-v1.0:0.8>',
         },
    15: {'label': '地铁',
         'prompt': 'e235,train interior,<lora:E235_V5:1>',
         },
    # 16: {'label': '试衣间',
    #      'prompt': 'fittingroom,<lora:fittingroomselfie:1>',
    #      },
    16: {'label': '秋天的童话',
         'prompt': 'rogowoarboretum,<lora:hjrogowoarboretum_v10:0.8>,beautiful tree with red and yellow leaves,sunny weather,natural lighting',
         },
    17: {'label': '日式房间',
         'prompt': 'ryokan,scenery,table,indoors,television,window,chair,cup,ceiling light,lamp,flower pot,sunlight,<lora:ryokan:0.8>',
         },
    18: {'label': '体育馆',
         'prompt': 'school gym,reflective floor,stage,scenery,indoors,wooden floor,<lora:school_gym_v0.1:1>',
         },
}


class MagicModel(object):
    operator = None
    sd_model_name = 'v1-5-pruned-emaonly.safetensors' if CONFIG['local'] else 'chilloutmix_NiPrunedFp32Fix-inpainting_zzg.inpainting'

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
        arge_idxs = {v: i for i, v in enumerate(['child', 'youth', 'middlescent'])}
        _age = arge_idxs[params['age']]
        viewpoint_mode_idxs = {v: i for i, v in enumerate(['front', 'side', 'back'])}
        _viewpoint_mode = viewpoint_mode_idxs[params['viewpoint_mode']]
        _model_mode = 0 if params['model_mode'] == 'normal' else 1  # 0:模特 ,1:人台
        # model and place type
        _model_type = int(params['model_type'])  # 模特类型
        _place_type = int(params['place_type'])  # 背景

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
                if self.operator.update_progress(10):
                    return {'success': True}

                # 切割衣服
                person0_box = [-1, -1, -1, -1]
                sam_images = []
                mask_images = []
                for dino_idx, dino_prompt in enumerate(_dino_clothing_text_prompt):
                    sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, dino_prompt,
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

                if self.operator.update_progress(20):
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
                    person_boxes, _ = self.operator.dino.dino_predict_internal(_input_image,
                                                                               self.operator.dino_model_name,
                                                                               "person",
                                                                               _box_threshold)
                    # sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, "person", 0.33, _input_image)
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

                if self.operator.update_progress(30):
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
                    if self.operator.update_progress(40):
                        return {'success': True}

        # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
        if self.operator.update_progress(50):
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
        sampler_index = 'DPM++ 2S a Karras'  # sampling method modules/sd_samplers_kdiffusion.py
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
        controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
            self.operator.cnet_idx+1].get_default_ui_unit()
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
        adetail_enabled = not self.operator.shared.cmd_opts.disable_adetailer
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
                    False,
                    # refiner
                    False, '', 0.8,
                    # seed
                    -1, False, -1, 0, 0, 0,
                    # soft inpainting
                    False, 1, 0.5, 4, 0, 0.5, 2,
                    # tiled global
                    False, 'DemoFusion', True, 128, 64, 4, 2, False, 10, 1, 1, 64, False, True, 3, 1, 1
                    ]

        # celery_task.update_state(state='PROGRESS', meta={'progress': 50})
        if self.operator.update_progress(50):
            return {'success': True}

        ok_img_count = 0
        fuck_img_count = 0
        ok_res = []
        ok_sam_res = []
        sam_bg_tmp_png_fp_list = []
        _output_model_width, _output_model_height = init_img.size
        # while ok_img_count < batch_size:
        # 模特生成
        res = self.operator.img2img.img2img(task_id, 4, sd_model_positive_prompt, sd_model_negative_prompt,
                                            prompt_styles,
                                            init_img,
                                            sketch,
                                            init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                            init_img_inpaint, init_mask_inpaint,
                                            steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                            n_iter,
                                            batch_size,
                                            cfg_scale,
                                            image_cfg_scale,
                                            denoising_strength,
                                            selected_scale_tab,
                                            _output_model_height,
                                            _output_model_width,
                                            scale_by,
                                            resize_mode,
                                            False,
                                            inpaint_full_res_padding,
                                            inpainting_mask_invert,
                                            '',
                                            '',
                                            '',
                                            override_settings_texts,
                                            False,
                                            [],
                                            '',
                                            *sam_args)

        self.operator.devices.torch_gc()
        for res_idx, res_img in enumerate(res[0][:_batch_size]):
            if getattr(res_img, 'already_saved_as', False):
                if self.operator.predict_image(res_img.already_saved_as):
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
                    sam_bg_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, 'person',
                                                                                0.3,
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
        if self.operator.update_progress(70):
            return {'success': True}

            # else:
        # 背景生成
        for ok_idx, ok_model_res in enumerate(ok_res):
            # cache_fp = f"tmp/model_only_{ok_idx}_{pic_name}_save.png"
            # ok_model_res.save(cache_fp)

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"
            steps = 20
            sampler_index = 'DPM++ 2S a Karras'  # sampling method modules/sd_samplers_kdiffusion.py
            inpainting_fill = 1
            restore_faces = False
            batch_size = 1
            resize_mode = 2  # 1: crop and resize 2: resize and fill
            inpaint_full_res = 0  # choices=["Whole picture", "Only masked"]
            inpaint_full_res_padding = 0
            inpainting_mask_invert = 1  # Mask mode 0: Inpaint masked - 1: Inpaint not masked
            cfg_scale = 9

            # controlnet args
            controlnet_args_unit1 = self.operator.scripts.scripts_img2img.alwayson_scripts[
                self.operator.cnet_idx+1].get_default_ui_unit()

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
                        False, sam_bg_tmp_png_fp_list[ok_idx], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
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
                        False,
                        # refiner
                        False, '', 0.8,
                        # seed
                        -1, False, -1, 0, 0, 0,
                        # soft inpainting
                        False, 1, 0.5, 4, 0, 0.5, 2,
                        # tiled global
                        False, 'DemoFusion', True, 128, 64, 4, 2, False, 10, 1, 1, 64, False, True, 3, 1, 1
                        ]
            ok_res[ok_idx] = \
                self.operator.img2img.img2img(task_id, 4, sd_bg_positive_prompt, sd_bg_negative_prompt,
                                              prompt_styles,
                                              ok_sam_res[ok_idx],
                                              sketch,
                                              init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                              init_img_inpaint, init_mask_inpaint,
                                              steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
                                              n_iter,
                                              batch_size,
                                              cfg_scale,
                                              image_cfg_scale,
                                              denoising_strength,
                                              selected_scale_tab,
                                              _output_model_height,
                                              _output_model_width,
                                              scale_by,
                                              resize_mode,
                                              inpaint_full_res,
                                              inpaint_full_res_padding,
                                              inpainting_mask_invert,
                                              '',
                                              '',
                                              '',
                                              override_settings_texts,
                                              False,
                                              [],
                                              '',
                                              *sam_args)[0][0]
            self.operator.devices.torch_gc()

        # celery_task.update_state(state='PROGRESS', meta={'progress': 90})
        if self.operator.update_progress(90):
            return {'success': True}
        #  -------------------------------------------------------------------------------------
        # storage img

        return ok_res

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
                '(middle-aged:1.3)',
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
