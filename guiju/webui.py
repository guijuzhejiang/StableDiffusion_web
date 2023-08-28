# coding=utf-8
# @Time : 2023/5/23 下午3:12
# @File : webui.py
import copy
import datetime
import io
import math
import os
import random
import re
import string
import traceback

import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
import gradio
import gradio as gr
import modules.scripts
from guiju.global_var import html_label
from guiju.segment_anything_util.dino import dino_model_list, dino_predict_internal
from guiju.segment_anything_util.sam import sam_model_list, sam_predict
from modules import shared, script_callbacks, devices, scripts_postprocessing, scripts, generation_parameters_copypaste, \
    images, ui_postprocessing
from modules.paths import script_path, data_path
import modules.img2img
from modules.shared import cmd_opts
from guiju.predictor_opennsfw2 import predict_image
import sys


def get_prompt(_gender, _age, _viewpoint, _model_mode=0):
    sd_positive_prompts_dict = OrderedDict({
        'gender': [
            # female
            '1girl',
            # male
            '1boy',
        ],
        'age': [
            # child
            f'(child:1.3){"" if _gender else ", <lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>"}',
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
            'full body' if _model_mode == 0 else '(full body:1.8)',
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


def show_prompt(_gender, _age, _viewpoint, _model_mode):
    _sd_positive_prompt, _sd_negative_prompt = get_prompt(_gender, _age, _viewpoint, _model_mode)
    return f'sd_positive_prompt: {_sd_positive_prompt}\n\nsd_negative_prompt: {_sd_negative_prompt}'


# def resize_rgba_image_pil_to_cv(image, target_ratio=0.5, quality=80):
#     # 将PIL RGBA图像转换为BGR图像
#     cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
#
#     # 获取原始图像的尺寸
#     original_height, original_width = cv_image.shape[:2]
#
#     # 计算原始图像的长宽比
#     original_ratio = original_width / original_height
#
#     # 计算应该添加的填充量
#     padded_image = cv_image
#     if original_ratio > target_ratio:
#         # 需要添加垂直填充
#         target_height = int(original_width / target_ratio)
#         # top = int((target_height - original_height) / 2)
#         # bottom = target_height - original_height - top
#         # padded_image = cv2.copyMakeBorder(cv_image, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
#         padded_image = cv2.copyMakeBorder(cv_image, int(target_height - original_height), 0, 0, 0, cv2.BORDER_REPLICATE)
#     else:
#         if original_width <= original_height:
#             # 需要添加水平填充
#             target_width = int(original_height * target_ratio)
#             left = int((target_width - original_width) / 2)
#             right = target_width - original_width - left
#             padded_image = cv2.copyMakeBorder(cv_image, 0, 0, left, right, cv2.BORDER_REPLICATE)
#
#     # 压缩图像质量
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#     _, jpeg_data = cv2.imencode('.jpg', padded_image, encode_param)
#
#     # 将压缩后的图像转换为PIL图像
#     pil_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')
#
#     return pil_image


def configure_image(image, person_pos, target_ratio=0.5, quality=90, padding=8):
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


def padding_rgba_image_pil_to_cv(original_image, pl, pr, pt, pb, person_box, padding=8):
    original_width, original_height = original_image.size
    edge_color = original_image.getpixel((padding, padding))
    # padded_image = Image.new('RGBA', (original_width + pl + pr, original_height + pt + pb), edge_color)
    # padded_image.paste(original_image, (pl, pt), mask=original_image)

    cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGBA2BGRA)
    h, w, _ = cv_image.shape

    padded_image = cv2.copyMakeBorder(cv_image[padding:h-padding, padding:w-padding],
                                      pt+padding if person_box[1] > 8 else 0,
                                      pb+padding if 8 <= original_height - person_box[3] else 0,
                                      pl+padding if person_box[0] > 8 else 0,
                                      pr+padding if 8 <= original_width - person_box[2] else 0,
                                      cv2.BORDER_REPLICATE)
    padded_image = cv2.copyMakeBorder(padded_image,
                                      pt+padding if person_box[1] <= 8 else 0,
                                      pb+padding if 8 > original_height - person_box[3] else 0,
                                      pl+padding if person_box[0] <= 8 else 0,
                                      pr+padding if 8 > original_width - person_box[2] else 0,
                                      cv2.BORDER_REPLICATE,
                                      # value=edge_color
                                      )
    padded_image = cv2.cvtColor(np.array(padded_image), cv2.COLOR_BGRA2RGBA)
    return padded_image


def crop_image(original_image, person_box):
    person_box = [int(x) for x in person_box]
    original_width, original_height = original_image.size
    cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGBA2BGRA)
    crop_image = cv_image[person_box[1]:person_box[3], person_box[0]:person_box[2]]
    crop_image = cv2.cvtColor(np.array(crop_image), cv2.COLOR_BGRA2RGBA)
    return crop_image


def create_ui():
    shared.state.server_command = None
    reload_javascript()
    # init sam
    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    # modules.scripts.scripts_img2img.alwayson_scripts[0].args_from = 1
    # modules.scripts.scripts_img2img.alwayson_scripts[0].args_to = 21

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

    # web ui
    with gr.Blocks(analytics_enabled=False, visible=False) as extras_interface:
        ui_postprocessing.create_ui()
    with gr.Blocks(analytics_enabled=False, title="cloths_inpaint", css='style.css') as demo:
        with gr.Row(elem_id='2nd_row', visible=False):
            lang_vals = list(html_label['lang_selection'].values())
            lang_sel_list = gr.Dropdown(label="language", elem_id="lang_list", choices=lang_vals, type="value",
                                        value=html_label['lang_selection'][shared.lang])

        with gr.Tabs(elem_id="tabs") as tabs:
            with gr.TabItem(html_label['ai_model_label'][shared.lang], elem_id="generate_ai_model_tab"):
                with gr.Row(elem_id=f"image_row"):
                    with gr.Column(scale=1):
                        input_image = gr.Image(label=html_label['input_image_label'][shared.lang],
                                               elem_id=f"input_image",
                                               source="upload",
                                               type="pil", image_mode="RGBA").style(height=640)

                    with gr.Column(scale=1):
                        with gr.Group(elem_id=f"gallery_container"):
                            result_gallery = gr.Gallery(label=html_label['output_gallery_label'][shared.lang],
                                                        elem_id=f"result_gallery").style(
                                columns=3,
                                rows=1,
                                preview=True,
                                height=640)
                        # .style(grid=3)

                # img2img input args
                with gr.Row(elem_id=f"control_row"):
                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Column(scale=10):
                                # batch_size = gr.Slider(minimum=1, maximum=3, step=1, label=html_label['batch_size_label'][shared.lang],
                                #                        value=1, elem_id="batch_size")
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        batch_size = gr.Dropdown(label=html_label['batch_size_label'][shared.lang],
                                                                 choices=[1, 2, 3],
                                                                 type='value', value='1', elem_id="batch_size")
                                    with gr.Column(scale=1):
                                        model_mode = gr.Radio(label=html_label['model_mode_label'][shared.lang],
                                                              choices=html_label['model_mode_list'][shared.lang],
                                                              value=html_label['model_mode_list'][shared.lang][0],
                                                              type="index", elem_id="model_mode", interactive=True,
                                                              visible=True)
                                    with gr.Column(scale=1):
                                        gender = gr.Radio(label=html_label['output_gender_label'][shared.lang],
                                                          choices=html_label['output_gender_list'][shared.lang],
                                                          value=html_label['output_gender_list'][shared.lang][0],
                                                          type="index", elem_id="gender")
                                    with gr.Column(scale=1):
                                        age = gr.Radio(label=html_label['output_age_label'][shared.lang],
                                                       choices=html_label['output_age_list'][shared.lang],
                                                       value=html_label['output_age_list'][shared.lang][1],
                                                       type="index", elem_id="age")

                                with gr.Row():
                                    viewpoint_mode = gr.Radio(label=html_label['output_viewpoint_label'][shared.lang],
                                                              choices=html_label['output_viewpoint_list'][shared.lang],
                                                              value=html_label['output_viewpoint_list'][shared.lang][0],
                                                              type="index", elem_id="viewpoint_mode", interactive=True,
                                                              visible=True)
                                    cloth_part = gr.CheckboxGroup(choices=html_label['input_part_list'][shared.lang],
                                                                  value=html_label['input_part_list'][shared.lang][:2],
                                                                  label=html_label['input_part_label'][shared.lang],
                                                                  elem_id="input_part",
                                                                  type="index",
                                                                  visible=False)

                            with gr.Column(scale=1):
                                regenerate = gr.Button(html_label['generate_btn_label'][shared.lang],
                                                       elem_id=f"re_generate",
                                                       variant='primary')
                                interrupt = gr.Button(html_label['interrupt_btn_label'][shared.lang],
                                                      elem_id=f"interrupt",
                                                      visible=False)
                                prompt = gr.Button('prompt', elem_id=f"show_prompt",
                                                   visible=True if cmd_opts.debug_mode else False)
                    with gr.Column(scale=1):
                        with gr.Row():
                            hint1 = gr.Text(value=html_label['hint1'][shared.lang], elem_id="hint1", label='',
                                            elem_classes='hint')
                        with gr.Row():
                            hint2 = gr.Text(value=html_label['hint2'][shared.lang], elem_id="hint2", label='',
                                            elem_classes='hint')

            with gr.TabItem(html_label['generate_hires_label'][shared.lang], elem_id="generate_hires_tab"):
                with gr.Row(elem_id=f"hires_image_row"):
                    with gr.Column(scale=1):
                        with gr.Group(elem_id=f"hires_gallery_container"):
                            hires_input_gallery = gr.Gallery(label=html_label['hires_input_gallery_label'][shared.lang],
                                                             elem_id=f"result_gallery").style(
                                columns=3,
                                rows=1,
                                preview=True,
                                height=640)
                    with gr.Column(scale=1):
                        hires_result_image = gr.Image(label=html_label['hires_result_image_label'][shared.lang],
                                                      elem_id=f"hires_input_image",
                                                      type="pil", image_mode="RGBA").style(height=640)

                with gr.Row(elem_id=f"hires_control_row"):
                    output_resolution = gr.Dropdown(label=html_label['output_resolution_label'][shared.lang],
                                                    elem_id="lang_list",
                                                    choices=html_label['output_resolution_list'], type="value",
                                                    value=html_label['output_resolution_list'][0])
                    choosing_index_4_hires = gr.Radio(
                        label=html_label['choosing_index_4_hires_label'][shared.lang],
                        choices=['0', '1', '2'],
                        value='0',
                        type="index", elem_id="choosing_index_4_hires",
                        visible=True)
                    generate_hires = gr.Button(html_label['generate_hires_label'][shared.lang],
                                               variant='primary',
                                               interactive=False,
                                               elem_id=f"generate_hires")

        with gr.Row(visible=True if cmd_opts.debug_mode else False):
            sam_result = gr.Text(value="", label="Status")

        # def cloth_partchange(_c):
        #     if 0 in _c:
        #         if len(_c) > 1:
        #             return [html_label['input_part_list'][shared.lang][x] for x in _c if x != 0]
        #         else:
        #             return [html_label['input_part_list'][shared.lang][0]]
        #     else:
        #         return [html_label['input_part_list'][shared.lang][x] for x in _c if x != 0]

        # cloth_part.change(fn=cloth_partchange,
        #                   inputs=[cloth_part],
        #                   outputs=[cloth_part])

        regenerate.click(
            fn=proceed_cloth_inpaint,
            _js='guiju_submit',
            inputs=[batch_size,
                    input_image,
                    gender,
                    age,
                    viewpoint_mode,
                    cloth_part,
                    model_mode,
                    ],
            outputs=[result_gallery, hires_input_gallery, choosing_index_4_hires, generate_hires, sam_result]
        )

        generate_hires.click(
            fn=proceed_generate_hires,
            _js='guiju_hires_submit',
            inputs=[hires_input_gallery,
                    choosing_index_4_hires,
                    output_resolution,
                    ],
            outputs=[hires_result_image]
        )

        def reload_ui(lang):
            for k, v in html_label['lang_selection'].items():
                if v == lang:
                    shared.lang = k
            print(lang)
            shared.state.request_restart()

        lang_sel_list.change(
            fn=reload_ui,
            _js='restart_reload2',
            inputs=[lang_sel_list],
        )

        interrupt.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

        prompt.click(
            fn=show_prompt,
            inputs=[gender, age, viewpoint_mode, model_mode],
            outputs=[sam_result],
        )

    modules.scripts.scripts_current = None
    script_callbacks.ui_settings_callback()

    return demo


def proceed_generate_hires(_hires_input_gallery, _choosing_index_4_hires, _output_resolution):
    _input_image = Image.open(_hires_input_gallery[_choosing_index_4_hires]['name'])
    if _input_image.mode != "RGBA":
        _input_image = _input_image.convert("RGBA")
    _input_image_width, _input_image_height = _input_image.size
    print('proceed_generate_hires')

    # 定义正则表达式模式，用于匹配宽度和高度
    pattern = r'(\d+)x(\d+)'
    # 使用 re.findall() 方法进行匹配
    matches = re.findall(pattern, _output_resolution)

    # 检查是否找到匹配结果
    if matches:
        # 匹配结果是一个元组列表，每个元组包含宽度和高度
        _output_width, _output_height = matches[0]
        _output_width = int(_output_width)
        _output_height = int(_output_height)
    else:
        raise Exception("未找到匹配的宽度和高度信息")

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
        controlnet_args_unit1 = modules.scripts.scripts_img2img.alwayson_scripts[cnet_idx].get_default_ui_unit()
        controlnet_args_unit1.batch_images = ''
        controlnet_args_unit1.control_mode = 'ControlNet is more important'
        controlnet_args_unit1.enabled = True
        controlnet_args_unit1.guidance_end = 1
        controlnet_args_unit1.guidance_start = 0  # ending control step
        controlnet_args_unit1.image = {'image':np.array(_input_image.convert('RGB')), 'mask': np.full((_input_image_height, _input_image_width, 4), (0, 0, 0, 255), dtype=np.uint8)}
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
        fake_args = {'ad_model': 'face_yolov8m.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
                     'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                     'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4,
                     'ad_denoising_strength': 0.4,
                     'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                     'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512,
                     'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
                     'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
                     'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                     'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1,
                     'is_api': ()}
        sam_args = [0,
                    adetail_enabled, fake_args, fake_args,  # adetail args
                    controlnet_args_unit1, controlnet_args_unit2, controlnet_args_unit3,  # controlnet args
                    False, False, 0, None,  # sam args
                    [], 0, False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [], False, 0, None, None,
                    '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True,
                    50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                    '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>',
                    128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False,
                    False, 'positive', 'comma', 0, False, False, '',
                    '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                    64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None, False, None,
                    None, False, None, None, False, 50
                    ]

        cnet_res = modules.img2img.img2img(task_id, 0, sd_positive_prompt, sd_negative_prompt, prompt_styles, init_img,
                                           sketch,
                                           init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                           init_img_inpaint, init_mask_inpaint,
                                           steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, restore_faces,
                                           tiling,
                                           n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed,
                                           subseed,
                                           subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras,
                                           selected_scale_tab, padding_height, padding_width, scale_by, resize_mode,
                                           inpaint_full_res,
                                           inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir,
                                           img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                           override_settings_texts,
                                           *sam_args)
    else:
        padding_height = _input_image_height
        padding_width = _input_image_width

    # extra upscaler
    cnet_res_img = _input_image if _output_ratio == 0.5 else cnet_res[0][0]
    scales = _output_width / padding_width
    args = (0, scales, padding_height, padding_width, True, 'R-ESRGAN 4x+', 'None', 0, 0, 0, 0)
    assert cnet_res_img, 'image not selected'

    devices.torch_gc()
    image_data = []
    image_names = []
    outputs = []
    image_data.append(cnet_res_img)
    image_names.append(None)
    outpath = shared.opts.outdir_samples or shared.opts.outdir_extras_samples

    for image, name in zip(image_data, image_names):
        existing_pnginfo = image.info or {}

        pp = scripts_postprocessing.PostprocessedImage(image.convert("RGB"))

        scripts.scripts_postproc.run(pp, args)

        if shared.opts.use_original_name_batch and name is not None:
            basename = os.path.splitext(os.path.basename(name))[0]
        else:
            basename = ''

        infotext = ", ".join(
            [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if
             v is not None])

        if shared.opts.enable_pnginfo:
            pp.image.info = existing_pnginfo
            pp.image.info["postprocessing"] = infotext

        # if save_output:
        images.save_image(pp.image, path=outpath, basename=basename, seed=None, prompt=None,
                          extension=shared.opts.samples_format, info=infotext, short_filename=True, no_prompt=True,
                          grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo,
                          forced_filename=None)

        outputs.append(pp.image)
    devices.torch_gc()

    return outputs[0]


def proceed_cloth_inpaint(_batch_size, _input_image, _gender, _age, _viewpoint_mode, _cloth_part, _model_mode):
    _batch_size = int(_batch_size)
    shared.state.interrupted = False

    output_height = 1024
    output_width = 512
    _dino_model_name = "GroundingDINO_SwinB (938MB)"
    _sam_model_name = 'samhq_vit_h_1b3123.pth'
    # _input_part_prompt = [['upper cloth'], ['pants', 'skirts'], ['shoes']]
    # _dino_text_prompt = ' . '.join([y for x in _cloth_part for y in _input_part_prompt[x]])
    # _dino_text_prompt = 'dress'
    _dino_clothing_text_prompt = 'clothing' if _model_mode == 1 else 'clothing . shorts'
    _box_threshold = 0.3

    if _input_image is None:
        return None, None
    else:
        origin_image_path = f'tmp/origin_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
        _input_image.save(origin_image_path, format='PNG')

        if predict_image(origin_image_path):
            raise gr.Error("The input image is NSFW!!!")
        try:
            _input_image_width, _input_image_height = _input_image.size

            # 得到的box宽长比小于0.5则resize到2/3，其余保持原本长宽比， 得到的缩放比例后左右不padding
            # real people
            if _model_mode == 0:
                person_boxes, _ = dino_predict_internal(_input_image, _dino_model_name, "person",
                                                        _box_threshold)
                person0_box = [int(x) for x in person_boxes[0]]
                person0_width = person0_box[2] - person0_box[0]
                person0_height = person0_box[3] - person0_box[1]
                _input_image = configure_image(_input_image, person_boxes[0], target_ratio=2/3 if (person0_width / person0_height) < 0.5 else person0_width / person0_height)

            # artificial model
            else:
                person_boxes, _ = dino_predict_internal(_input_image, _dino_model_name, _dino_clothing_text_prompt, _box_threshold)

                # get max area clothing box
                x_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 0 or i ==2]
                y_list = [int(y) for x in person_boxes for i, y in enumerate(x) if i == 1 or i ==3]
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
                padding_top = int(person0_height * top_ratio - int(person0_box[1])) if (int(person0_box[1]) / person0_height) < top_ratio else 0
                padding_bottom = int(person0_height * bottom_ratio - (_input_image_height - int(person0_box[3]))) if ((_input_image_height - int(person0_box[3])) / person0_height) < bottom_ratio else 0

                new_person0_box = []

                _input_image = padding_rgba_image_pil_to_cv(_input_image, padding_left, padding_right, padding_top, padding_bottom, person0_box)
                # _input_image = configure_image(_input_image, [0, 0, padding_left + _input_image_width + padding_right,
                #                                               padding_top + _input_image_height + padding_bottom],
                #                                target_ratio=output_width / output_height)
                _input_image = configure_image(_input_image,
                                               [0 if (int(person0_box[0]) / person0_width) < left_ratio else person0_box[0] - int(person0_width * left_ratio),
                                                padding_top if padding_top > 0 else person0_box[1] - int(person0_height * top_ratio),
                                                _input_image_width if ((_input_image_width - int(person0_box[2])) / person0_width) < right_ratio else padding_left + person0_box[2] + int(person0_width * right_ratio),
                                                _input_image_height + padding_bottom + padding_top if padding_bottom > 0 else padding_top + person0_box[3] + int(person0_height * bottom_ratio)],
                                               target_ratio=2/3 if (person0_width / person0_height) < 0.5 else person0_width / person0_height)

        except Exception:
            print(traceback.format_exc())
            print('preprocess img error')
            raise gr.Error("preprocess img error")
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
            _, jpeg_data = cv2.imencode('.png', cv2.cvtColor(np.array(_input_image), cv2.COLOR_RGBA2BGRA), encode_param)

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
    for idx, sam_mask_img in enumerate(sam_result_gallery):
        cache_fp = f"tmp/{idx}_{pic_name}.png"
        sam_mask_img.save(cache_fp)
        sam_result_tmp_png_fp.append({'name': cache_fp})

    task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

    sd_positive_prompt, sd_negative_prompt = get_prompt(_gender, _age, _viewpoint_mode, _model_mode)

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
    face_args = {'ad_model': 'face_yolov8m.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
                 'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4,
                 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512,
                 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
                 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
                 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                 'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1,
                 'is_api': ()}
    hand_args = {'ad_model': 'hand_yolov8s.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
                 'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0,
                 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4,
                 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
                 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512,
                 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
                 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
                 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
                 'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1,
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
                128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'],
                False, False, 'positive', 'comma', 0, False, False, '',
                '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None, False, None, None,
                False, None, None, False, 50
                ]

    try:
        ok_img_count = 0
        fuck_img_count = 0
        ok_res = []
        while ok_img_count < batch_size:
            res = modules.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt, prompt_styles, init_img,
                                          sketch,
                                          init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                          init_img_inpaint, init_mask_inpaint,
                                          steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, restore_faces,
                                          tiling,
                                          n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed,
                                          subseed,
                                          subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras,
                                          selected_scale_tab, height, width, scale_by, resize_mode, inpaint_full_res,
                                          inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir,
                                          img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                          override_settings_texts,
                                          *sam_args)
            for res_img in res[0]:
                if getattr(res_img, 'already_saved_as', False):
                    if predict_image(res_img.already_saved_as):
                        fuck_img_count += 1
                        if fuck_img_count > 10:
                            raise gr.Error("fatal error")
                    else:
                        ok_img_count += 1
                        ok_res.append(res_img)
    except Exception:
        raise gr.Error("found no cloth")

    return ok_res, ok_res, gr.Radio.update(choices=[str(x) for x in range(len(res[0]))], value=0), gr.Button.update(
        interactive=True), 'done.'


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def javascript_html():
    # Ensure localization is in `window` before scripts
    # head = f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'
    head = ''
    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    for script in modules.scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    return head


def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        head += stylesheet(cssfile)

    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))

    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
