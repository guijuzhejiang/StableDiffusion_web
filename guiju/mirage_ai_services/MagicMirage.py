# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import copy
import datetime
import glob
import os
import random
import string

import numpy as np
from PIL import Image

from lora_config import reference_dir
from modules.sd_samplers_kdiffusion import samplers_k_diffusion

lora_mirage_dict = {
    12: {'label': '侏罗纪',
         'prompt': '<lora:侏罗纪花园_v1.0:0.7>,horror,deep shadow,large Tyrannosaurus Rex,forest,trees,huge waterfall,river',
         },
    1: {'label': '巨大战舰',
        'prompt': '<lora:neowrsk_v2:0.7>,<lora:[LoHa] Octans八分儀 Stylev2:1>,Megalophobia,giant phobia,cloud,low angle,chaosmix,chaos,horror,neowrsk,octans,flying spacecraft,floating in the sky,spaceship,cyberpunk aesthetics,electrical storm,plasma turret fire,interstellar warfare,tension,decaying space station backdrop,ominous,nebula-filled cosmos,(huge:1.5),from below,red theme,mysterious,ethereal,sharp focus,hot pinks,and glowing purples,(giant clothes),Dramatic Lighting,Bold Coloration,Vibrant Warmth,deep shadow,astonishing level of detail,Horizon composition,universe,Hal\'s Mobile Castle,Huge Sky Castle',
        },
    2: {'label': '恐怖死神',
        'prompt': '<lora:Ghost Concept_v1.0:0.7>,visually stunning,elegant,incredible details,g0s1,faceless,no humans,cloak,robe,torn clothes,torn fabric,floating,grim reaper,black reaper,fantasy theme,horror \(theme\),scythe,holding scythe,death,ghost,hood',
        },
    3: {'label': '星际大战',
        'prompt': '<lora:末日-宇宙（场景）_v1.0:0.6>,horror,(A huge spaceship:1.5),(solo:1.5),(A rectangular spacecraft resembling the shape of an aircraft carrier:1.5),Full of art,Cosmic galaxy background,Doomsday scenario,Crumbling earth,a volcano erupts,energy blast,The fleeing spaceship',
        },
    4: {'label': '月下城堡',
        'prompt': '<lora:Ancient_city:1>,BJ_Ancient_city,outdoors,sky,cloud,water,tree,moon,fire,building,scenery,full_moon,stairs,mountain,architecture,east_asian_architecture,cinematic lighting,morning red,abundant,wallpaper,huge bridges',
        },
    5: {'label': '未来机器城',
        'prompt': '<lora:XSArchi_127:1>,<lora:Concept_scenery_background:0.3>,solo,(zenithal angle),sunset,(by Iwan Baan),skyscraper,japan style,arasaka tower,neon lights,cyberpunk,cyberpunk series,steam power,ultra-wide angle',
        },
    6: {'label': '机甲怪兽',
        'prompt': '<lora:机甲怪兽风格lora_v1.0:0.5>,monster,Alien monsters invade Earth,Dragon-shaped monster,huge,thriller,robot,Mecha,chilling,horrifying,terrifying',
        },
    8: {'label': '浮岛宝塔',
        'prompt': '<lora:(LIb首发)CG古风大场景类_v2.0:1>,Unreal Engine 5,CG,abg,Chinese CG scene,top down,scenery,waterfall,cloud,tree,architecture,sky,outdoors,floating island,day,east asian architecture,fantasy,mountain,water,bridge,pagoda,castle,building,blue sky,tower,fog',
        },
    7: {'label': '满月与海',
        'prompt': '<lora:满月与大海_v0.1:1>,ambient lighting,professional artwork,Ambient Occlusion,surrealism,illusion,only sky,unreal,depth of field,focus to the sky,silver theme,lunarYW,sea',
        },
    9: {'label': '满月古城',
        'prompt': '<lora:(LIb首发)CG古风大场景类_v2.0:1>,HD,cg,Chinese CG scene,unreal 5 engine,Mid-autumn,full moon,night view,plants,ancient buildings,bridge,Backlight,Creek,Clouds,Chinese architecture,brightly lit',
        },
    10: {'label': '科幻世界',
         'prompt': '<lora:新科幻Neo Sci-Fi_v1.0:1>,sci-fi city,modern architecture style,river,a floating city in the sky,super high-rise building,high resolution,outdoor,(day:1.2),(blue sky:1.3),water,soft lighting,(dramatic scene),(Epic composition:1.2)',
         },
    11: {'label': '蓬莱仙岛',
         'prompt': '<lora:(LIb首发)CG古风大场景类_v2.0:1>,HD,cg,Chinese CG scene,unreal 5 engine,floating island,abg,large waterfall left in the middle,rain,huge peaks',
         },
    0: {'label': '赛博朋克',
        'prompt': '<lora:Cyberpunk sceneV1:0.7>,Megalophobia,giant phobia,cloud,low angle,chaosmix,chaos,horror,tooimage cyberpunk futuristic city,flying spacecraft,floating in the sky,spaceship,cyberpunk aesthetics,electrical storm,plasma turret fire,interstellar warfare,tension,decaying space station backdrop,ominous,nebula-filled cosmos,(huge:1.5),from below,red theme,mysterious,ethereal,sharp focus,hot pinks,and glowing purples,(giant clothes),Dramatic Lighting,Bold Coloration,Vibrant Warmth,deep shadow,astonishing level of detail,Horizon composition,universe,Hal\'s Mobile Castle,Huge Sky Castle',
        },
}


class MagicMirage(object):
    operator = None
    sd_model_name = 'dreamshaper_8'

    denoising_strength_min = 0.45
    denoising_strength_max = 0.56

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

        # params
        _batch_size = int(params['batch_size'])
        _selected_place = int(params['place'])
        _sim = float(params['sim'])

        _input_image_width, _input_image_height = _input_image.size

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
        sam_result, person_boxes = self.operator.sam.sam_predict(self.operator.dino_model_name, 'person.bag',
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

        if self.operator.update_progress(20):
            return {'success': True}

        denoising_strength = (1 - _sim) * (self.denoising_strength_max - self.denoising_strength_min) + self.denoising_strength_min
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
        # denoising_strength = 0.75
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

        _reference_image = Image.open(_reference_image_path)
        _reference_image_w, _reference_image_h = _reference_image.size
        _reference_img_rgb_ndarray = np.array(_reference_image)
        _reference_img_mask_ndarray = np.zeros(shape=_reference_img_rgb_ndarray.shape)

        # 计算缩放比例，使mask_image适应background_image
        width_ratio = _reference_image.width / mask_image.width
        height_ratio = _reference_image.height / mask_image.height
        min_ratio = min(width_ratio, height_ratio)
        new_width = int(mask_image.width * min_ratio)
        new_height = int(mask_image.height * min_ratio)
        # 缩放mask_image
        resized_mask = mask_image.resize((new_width, new_height))
        # 计算粘贴位置
        paste_position = (
            (_reference_image.width - resized_mask.width) // 2,
            (_reference_image.height - resized_mask.height) // 2
        )
        # 创建一个透明度通道（alpha channel）的合成图像
        composite_ref_image = Image.new("RGB", _reference_image.size, (0, 0, 0))
        # composite_ref_image.paste(_reference_image, (0, 0))
        composite_ref_image.paste(resized_mask, paste_position, resized_mask)
        composite_ref_image = composite_ref_image.convert('1')

        controlnet_args_unit1.image = {
            'image': _reference_img_rgb_ndarray,
            'mask': np.array(composite_ref_image)*1,
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
        if self.operator.update_progress(50):
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
        res = self.operator.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt,
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

        return res
