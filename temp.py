# coding=utf-8
# @Time : 2023/5/30 上午10:50
# @File : temp.py
import torch
import os
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import requests
from PIL import Image
from io import BytesIO


def show_images(imgs, rows=1, cols=3):
    assert len(imgs) == rows * cols
    w_ori, h_ori = imgs[0].size
    for img in imgs:
        w_new, h_new = img.size
        if w_new != w_ori or h_new != h_ori:
            w_ori = max(w_ori, w_new)
            h_ori = max(h_ori, h_new)

    grid = Image.new('RGB', size=(cols * w_ori, rows * h_ori))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w_ori, i // cols * h_ori))
    return grid


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "cloth"
# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
# image = Image.open('/home/ray/Workspace/project/stable_diffusion/clothing_inpaint/tmp/JJebLrVFNMTPydM.png')
# mask_image = Image.open('/home/ray/Workspace/project/stable_diffusion/clothing_inpaint/tmp/iDTZEybZURwsEXv.png')

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

image.save("./yellow_cat_on_park_bench.png")
from diffusers import StableDiffusionInpaintPipeline

a = (0, True, {'ad_model': 'face_yolov8m.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
               'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0, 'ad_dilate_erode': 4,
               'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4,
               'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32,
               'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512,
               'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
               'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
               'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious',
               'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1,
               'is_api': ()},
     {'ad_model': 'hand_yolov8s.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
      'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0, 'ad_dilate_erode': 4,
      'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4, 'ad_inpaint_only_masked': True,
      'ad_inpaint_only_masked_padding': 32, 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512,
      'ad_inpaint_height': 512, 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7,
      'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_restore_face': False,
      'ad_controlnet_model': 'None', 'ad_controlnet_module': 'inpaint_global_harmonious', 'ad_controlnet_weight': 1,
      'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1, 'is_api': ()},
     ' < scripts.controlnet_ui.controlnet_ui_group.UiControlNetUnit object at 0x7f2c5b053400 >',
     True, False, 0, ' < PIL.Image.Image image mode=RGBA size=738x1476 at 0x7F2C60A71D00 >',
     [{'name': 'tmp/MOzHrUHWmgzUdGp.png'}, {'name': 'tmp/vqzHInjxZzgMZAi.png'}, {'name': 'tmp/uJIAkDyuofUngKt.png'}],
     -2, False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
     '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0,
     False, 4, 0.5, 'Linear', 'None',
     '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.7</p>',
     128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False,
     'positive', 'comma', 0, False, False, '',
     '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
     64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0)

batch_images = {str} ''
control_mode = {str} 'Balanced'
enabled = {bool} True
guidance_end = {int} 1
guidance_start = {int} 0
image = {NoneType} None
input_mode = {InputMode} InputMode.SIMPLE
is_ui = {bool} True
loopback = {bool} False
low_vram = {bool} False
model = {str} 'control_v11p_sd15_normalbae'
module = {str} 'normal_bae'
output_dir = {str} ''
pixel_perfect = {bool} True
processor_res = {int} 512
resize_mode = {str} 'Crop and Resize'
threshold_a = {int} 64
threshold_b = {int} 64
weight = {int} 1