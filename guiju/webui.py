# coding=utf-8
# @Time : 2023/5/23 下午3:12
# @File : webui.py
import random
import string

import gradio as gr
import modules.scripts
from guiju.segment_anything_util.dino import dino_model_list
from guiju.segment_anything_util.sam import sam_model_list, sam_predict
import modules.img2img


def proceed_cloth_inpaint(_batch_size, _gender, _input_image):
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

    if _gender == 'female':
        sd_positive_prompt = '(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, an extremely delicate and beautiful, extremely detailed, 2k wallpaper, light smile, extremely detailed CG unity 8k wallpaper, ultra-detailed, highres, beautiful detailed girl, detailed fingers, light on face, 1girl, cute, young, realistic face, realistic body, girl posing for a photo,   good hand,looking at viewer, (simple background:1.3), (white background:1.3)'
        lora_positive_prompt = ',<lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>'
        lora_positive_prompt += lora_positive_prompt
        sd_negative_prompt = 'NSFW,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many'
    else:
        sd_positive_prompt = '(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, an extremely delicate and beautiful, extremely detailed, CG, unity , 2k wallpaper, Amazing, finely detail, light smile, extremely detailed CG unity 8k wallpaper, ultra-detailed, highres, detailed fingers, 1boy, young, realistic face, realistic body, good hand,(simple background:1.3), (white background:1.3),'
        lora_positive_prompt = ',<lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>'
        lora_positive_prompt += lora_positive_prompt
        sd_negative_prompt = 'NSFW,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many'

    prompt_styles = None
    init_img = None
    sketch = None
    init_img_with_mask = None
    inpaint_color_sketch = None
    inpaint_color_sketch_orig = None
    init_img_inpaint = None
    init_mask_inpaint = None
    steps = 40
    sampler_index = 0 # sampling method modules/sd_samplers_kdiffusion.py
    mask_blur = 4
    mask_alpha = 0
    inpainting_fill = 1
    restore_faces = True
    tiling = False
    n_iter = 1
    batch_size = 1
    cfg_scale = 7
    image_cfg_scale = 1.5
    denoising_strength = 0.90
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
    inpaint_full_res = 0
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 1
    img2img_batch_input_dir = ''
    img2img_batch_output_dir = ''
    img2img_batch_inpaint_mask_dir = ''
    override_settings_texts = []

    sam_args = [0, True, False, 0, _input_image,
                sam_result_tmp_png_fp,
                2, False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>',
                128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'],
                False, False, 'positive', 'comma', 0, False, False, '',
                '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0
                ]

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
    # init sam

    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    modules.scripts.scripts_img2img.alwayson_scripts[0].args_from = 1
    modules.scripts.scripts_img2img.alwayson_scripts[0].args_to = 21

    # web ui
    with gr.Blocks(analytics_enabled=False, title="cloths_inpaint", css='style.css') as demo:
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Image for Segment Anything", elem_id=f"input_image", source="upload",
                                       type="pil", image_mode="RGBA").style(height=640)

            with gr.Column(scale=1):
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"result_gallery").style(grid=3, height='100%', container=True)
                # .style(grid=3)

        # img2img input args
        with gr.Row():
            with gr.Column(scale=1):
                # batch_size = gr.Dropdown(choices=[1, 2, 3], value=1, label='Batch size',
                #                          elem_id="img2img_batch_size")
                batch_size = gr.Slider(minimum=1, maximum=3, step=1, label='Batch size', value=1, elem_id="batch_size")

            with gr.Column(scale=1):
                gender = gr.Radio(label='Output gender', choices=['male', 'female'], value='male',
                                  type="index", elem_id="gender")
            with gr.Column(scale=4):
                sam_result = gr.Text(value="", label="Status")

        input_image.change(
            fn=proceed_cloth_inpaint,
            inputs=[batch_size,
                    gender,
                    input_image],
            outputs=[result_gallery, sam_result]
        )

    modules.scripts.scripts_current = None

    return demo
