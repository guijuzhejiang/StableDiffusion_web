# coding=utf-8
# @Time : 2023/5/23 下午3:12
# @File : webui.py
import datetime
import io
import os
import random
import string

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import gradio
import gradio as gr
import modules.scripts
from guiju.global_var import html_label
from guiju.segment_anything_util.dino import dino_model_list
from guiju.segment_anything_util.sam import sam_model_list, sam_predict
from modules import shared, script_callbacks
from modules.paths import script_path, data_path
import modules.img2img
from modules.shared import cmd_opts


def get_prompt(_gender, _age, _viewpoint):
    age_prompts = ['child', 'youth', 'middlescent']
    if _gender == 0:
        sd_positive_prompt = f"(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, an extremely delicate and beautiful, extremely detailed, CG, unity , 2k wallpaper, Amazing, finely detail, extremely detailed CG unity 8k wallpaper, ultra-detailed, highres, beautiful detailed girl, {age_prompts[_age]}, detailed fingers, 1girl, young, realistic body, fluffy black hair, girl posing for a photo, good hand, (simple background:1.3), (white background:1.3),(full body:1.5), light smile, beautiful detailed nose, beautiful detailed eyes, long eyelashes"
        if _age != 2:
            sd_positive_prompt += ',<lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>'
    else:
        sd_positive_prompt = f'(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, an extremely delicate, extremely detailed, CG, unity , 2k wallpaper, Amazing, finely detail, extremely detailed CG unity 8k wallpaper, ultra-detailed, highres, (1boy:1.3), realistic body, (simple background:1.3), (white background:1.3), {age_prompts[_age]}, (full body:1.3), detailed nose, detailed eyes'

    if _viewpoint == 0:
        sd_positive_prompt += ', realistic face, extremely detailed eyes and face, light on face, looking at viewer'
    elif _viewpoint == 1:
        sd_positive_prompt += ',(side face:1.5), side view of face, lateral face, looking to the side'
    else:
        sd_positive_prompt += ',looking back'
    sd_negative_prompt = '(extra clothes:1.5),(clothes:1.5),(NSFW:1.3),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8), aged up, old fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, (bad_prompt_version2:0.8)'

    return sd_positive_prompt, sd_negative_prompt


def show_prompt(_gender, _age, _viewpoint):
    _sd_positive_prompt, _sd_negative_prompt = get_prompt(_gender, _age, _viewpoint)
    return f'sd_positive_prompt: {_sd_positive_prompt}\n\nsd_negative_prompt: {_sd_negative_prompt}'


def resize_rgba_image_pil_to_cv(image, target_ratio=0.5, quality=80):
    # 将PIL RGBA图像转换为BGR图像
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

    # 获取原始图像的尺寸
    original_height, original_width = cv_image.shape[:2]

    # 计算原始图像的长宽比
    original_ratio = original_width / original_height

    # 计算应该添加的填充量
    if original_ratio > target_ratio:
        # 需要添加垂直填充
        target_height = int(original_width / target_ratio)
        top = int((target_height - original_height) / 2)
        bottom = target_height - original_height - top
        padded_image = cv2.copyMakeBorder(cv_image, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
    else:
        # 需要添加水平填充
        target_width = int(original_height * target_ratio)
        left = int((target_width - original_width) / 2)
        right = target_width - original_width - left
        padded_image = cv2.copyMakeBorder(cv_image, 0, 0, left, right, cv2.BORDER_REPLICATE)

    # 压缩图像质量
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, jpeg_data = cv2.imencode('.jpg', padded_image, encode_param)

    # 将压缩后的图像转换为PIL图像
    pil_image = Image.open(io.BytesIO(jpeg_data)).convert('RGBA')

    return pil_image
# def pad_and_compress_rgba_image(original_image, target_ratio=0.5, fill_color=(0, 0, 0, 0), quality=80):
#     original_width, original_height = original_image.size
#
#     # 计算原始图像的长宽比
#     original_ratio = original_width / original_height
#
#     # 计算应该添加的填充量
#     if original_ratio > target_ratio:
#         # 需要添加垂直填充
#         target_height = original_width / target_ratio
#         pad_height = int((target_height - original_height) / 2)
#         pad_width = 0
#     else:
#         # 需要添加水平填充
#         target_width = original_height * target_ratio
#         pad_width = int((target_width - original_width) / 2)
#         pad_height = 0
#
#     # 获取原图的边缘颜色
#     edge_color = original_image.getpixel((0, 0))
#
#     # 创建新的空白图像并粘贴原始图像
#     padded_image = Image.new('RGBA', (original_width + 2 * pad_width, original_height + 2 * pad_height), edge_color)
#     padded_image.paste(original_image, (pad_width, pad_height), mask=original_image)
#
#     # 压缩图像质量并返回图像数据
#     output_buffer = BytesIO()
#     padded_image.save(output_buffer, format='PNG', quality=quality)
#     output_buffer.seek(0)
#
#     # 使用 PIL 的 Image.open() 函数加载图像数据
#     compressed_image = Image.open(output_buffer)
#
#     # 返回填充和压缩后的图像
#     return compressed_image


def proceed_cloth_inpaint(_batch_size, _input_image, _gender, _age, _viewpoint_mode):
    shared.state.interrupted = False
    if _input_image is None:
        return None, None
    else:
        _input_image.save(f'tmp/origin_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', format='PNG')
        _input_image = resize_rgba_image_pil_to_cv(_input_image)
        # _input_image.save(f'tmp/dddd_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', format='PNG')

    _sam_model_name = sam_model_list[0]
    _dino_model_name = dino_model_list[1]
    _dino_text_prompt = 'clothing'
    _box_threshold = 0.3
    sam_result_tmp_png_fp = []

    sam_result_gallery, sam_result = sam_predict(_dino_model_name, _dino_text_prompt,
                                                 _box_threshold,
                                                 _input_image)

    for sam_mask_img in sam_result_gallery:
        cache_fp = f"tmp/{''.join([random.choice(string.ascii_letters) for c in range(15)])}.png"
        sam_mask_img.save(cache_fp)
        sam_result_tmp_png_fp.append({'name': cache_fp})

    task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

    sd_positive_prompt, sd_negative_prompt = get_prompt(_gender, _age, _viewpoint_mode)

    prompt_styles = None
    init_img = _input_image
    sketch = None
    init_img_with_mask = None
    inpaint_color_sketch = None
    inpaint_color_sketch_orig = None
    init_img_inpaint = None
    init_mask_inpaint = None
    steps = 20
    sampler_index = 0  # sampling method modules/sd_samplers_kdiffusion.py
    mask_blur = 4
    mask_alpha = 0
    inpainting_fill = 1
    restore_faces = True
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
    height = 1024
    width = 512
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
    # cnet_idx = 1
    # controlnet_args = modules.scripts.scripts_img2img.alwayson_scripts[cnet_idx].get_default_ui_unit()
    # controlnet_args.batch_images = ''
    # controlnet_args.control_mode = 'My prompt is more important'
    # # controlnet_args.enabled = True if _controlnet_mode > 0 else False
    # controlnet_args.enabled = False
    # controlnet_args.guidance_end = 1
    # controlnet_args.guidance_start = 0
    # controlnet_args.image = None
    # # controlnet_args.input_mode = batch_hijack.InputMode.SIMPLE
    # controlnet_args.low_vram = False
    # controlnet_args.model = 'control_v11p_sd15_normalbae'
    # controlnet_args.module = 'normal_bae'
    # controlnet_args.pixel_perfect = True
    # controlnet_args.resize_mode = 'Resize and Fill'
    # controlnet_args.processor_res = 512
    # controlnet_args.threshold_a = 64
    # controlnet_args.threshold_b = 64
    # # controlnet_args.weight = 0.4 if _controlnet_mode == 1 else 1
    # controlnet_args.weight = 0.4

    # sam
    sam_args = [0, True, False, 0, _input_image,
                sam_result_tmp_png_fp,
                0,  # sam_output_chosen_mask
                False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                f'<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: {denoising_strength}</p>',
                128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'],
                False, False, 'positive', 'comma', 0, False, False, '',
                '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0
                ]
    # sam_args = [0,
    #             controlnet_args,  # controlnet args
    #             True, False, 0, _input_image,
    #             sam_result_tmp_png_fp,
    #             0 - 2,  # sam_output_chosen_mask
    #             False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
    #             '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
    #             True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
    #             f'<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: {denoising_strength}</p>',
    #             128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'],
    #             False, False, 'positive', 'comma', 0, False, False, '',
    #             '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
    #             64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0
    #             ]

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

    return res[0], 'done.'


def create_ui():
    shared.state.server_command = None
    reload_javascript()
    # init sam
    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    modules.scripts.scripts_img2img.alwayson_scripts[0].args_from = 1
    modules.scripts.scripts_img2img.alwayson_scripts[0].args_to = 21

    # cnet_idx = 0
    # sam_idx = 1
    # modules.scripts.scripts_img2img.alwayson_scripts[cnet_idx], modules.scripts.scripts_img2img.alwayson_scripts[
    #     sam_idx] = modules.scripts.scripts_img2img.alwayson_scripts[sam_idx], \
    #                modules.scripts.scripts_img2img.alwayson_scripts[cnet_idx]

    # sam
    # modules.scripts.scripts_img2img.alwayson_scripts[0].args_from = 2
    # modules.scripts.scripts_img2img.alwayson_scripts[0].args_to = 22
    #
    # # controlnet
    # modules.scripts.scripts_img2img.alwayson_scripts[1].args_from = 1
    # modules.scripts.scripts_img2img.alwayson_scripts[1].args_to = 2

    # web ui
    with gr.Blocks(analytics_enabled=False, title="cloths_inpaint", css='style.css') as demo:
        # with gr.Row(elem_id='1st_row'):
        #     gr.Label(visible=False)
        with gr.Row(elem_id='2nd_row', visible=False):
            lang_vals = list(html_label['lang_selection'].values())
            lang_sel_list = gr.Dropdown(label="language", elem_id="lang_list", choices=lang_vals, type="value",
                                        value=html_label['lang_selection'][shared.lang])
        with gr.Row(elem_id=f"image_row"):
            with gr.Column(scale=1):
                input_image = gr.Image(label=html_label['input_image_label'][shared.lang], elem_id=f"input_image",
                                       source="upload",
                                       type="pil", image_mode="RGBA").style(height=640)

            with gr.Column(scale=1):
                with gr.Group(elem_id=f"gallery_container"):
                    result_gallery = gr.Gallery(label=html_label['output_gallery_label'][shared.lang], show_label=False,
                                                elem_id=f"result_gallery").style(
                        columns=3,
                        rows=1,
                        preview=True,
                        height=640)
                # .style(grid=3)

        # img2img input args
        with gr.Row(elem_id=f"control_row"):
            # batch_size = gr.Dropdown(choices=[1, 2, 3], value=1, label='Batch size',
            #                          elem_id="img2img_batch_size")
            with gr.Column(scale=1):
                batch_size = gr.Slider(minimum=1, maximum=3, step=1, label=html_label['batch_size_label'][shared.lang],
                                       value=1, elem_id="batch_size")

            with gr.Column(scale=6):
                gender = gr.Radio(label=html_label['output_gender_label'][shared.lang],
                                  choices=html_label['output_gender_list'][shared.lang],
                                  value=html_label['output_gender_list'][shared.lang][0],
                                  type="index", elem_id="gender")
            with gr.Column(scale=6):
                age = gr.Radio(label=html_label['output_age_label'][shared.lang],
                               choices=html_label['output_age_list'][shared.lang],
                               value=html_label['output_age_list'][shared.lang][1],
                               type="index", elem_id="age")
            with gr.Column(scale=6):
                viewpoint_mode = gr.Radio(label=html_label['output_viewpoint_label'][shared.lang],
                                          choices=html_label['output_viewpoint_list'][shared.lang],
                                          value=html_label['output_viewpoint_list'][shared.lang][0],
                                          type="index", elem_id="viewpoint_mode", interactive=False, visible=False)
            with gr.Column(scale=1):
                regenerate = gr.Button(html_label['generate_btn_label'][shared.lang], elem_id=f"re_generate",
                                       variant='primary')
                interrupt = gr.Button(html_label['interrupt_btn_label'][shared.lang], elem_id=f"interrupt",
                                      visible=False)
                prompt = gr.Button('prompt', elem_id=f"show_prompt", visible=True if cmd_opts.debug_mode else False)

        with gr.Row():
            with gr.Column(scale=1):
                hint1 = gr.Text(value=html_label['hint1'][shared.lang], elem_id="hint1", label='', elem_classes='hint')
            with gr.Column(scale=1):
                hint2 = gr.Text(value=html_label['hint2'][shared.lang], elem_id="hint2", label='', elem_classes='hint')
        with gr.Row(visible=True if cmd_opts.debug_mode else False):
            sam_result = gr.Text(value="", label="Status")

        regenerate.click(
            fn=proceed_cloth_inpaint,
            _js='guiju_submit',
            inputs=[batch_size,
                    input_image,
                    gender,
                    age,
                    viewpoint_mode,
                    ],
            outputs=[result_gallery, sam_result]
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
            inputs=[gender, age, viewpoint_mode],
            outputs=[sam_result],
        )

    modules.scripts.scripts_current = None
    script_callbacks.ui_settings_callback()
    return demo


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
