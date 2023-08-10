# -*- encoding: utf-8 -*-
'''
@File    :   ocr_process.py    
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/21 上午10:21   ray      1.0         None
'''

import datetime
import os
import traceback
from lib.common.common_util import logging
from lib.redis_pipline.operator import Operator
from modules.devices import torch_gc, device
from modules.safe import unsafe_torch_load, load
from segment_anything import SamPredictor, sam_model_registry
import copy
import gc
from collections import OrderedDict
from PIL import Image
import numpy as np
import gradio as gr
import torch
from scipy.ndimage import label
from guiju.segment_anything_util.dino import show_boxes, dino_predict_internal, dino_install_issue_text


class OperatorOCR(Operator):
    num = 2
    cache = True
    cuda = True

    def __init__(self):
        Operator.__init__(self)
        """ load sam model """
        self.sam_model_cache = OrderedDict()
        self.sam_model_dir = 'extensions/sd-webui-segment-anything/models/sam'
        self.sam_model_list = [f for f in os.listdir(self.sam_model_dir) if os.path.isfile(os.path.join(self.sam_model_dir, f)) and f.split('.')[-1] != 'txt']
        self.sam_model = self.init_sam_model(self.sam_model_list[0])
        print('SAM model is Initialized')

    def garbage_collect(self):
        gc.collect()
        torch_gc()

    def clear_sam_cache(self):
        self.sam_model_cache.clear()
        gc.collect()
        torch_gc()

    def init_sam_model(self, sam_model_name):
        print("Initializing SAM")
        if sam_model_name in self.sam_model_cache:
            sam = self.sam_model_cache[sam_model_name]
            return sam
        elif sam_model_name in self.sam_model_list:
            self.clear_sam_cache()
            self.sam_model_cache[sam_model_name] = self.load_sam_model(sam_model_name)
            return self.sam_model_cache[sam_model_name]
        else:
            Exception(
                f"{sam_model_name} not found, please download model to models/sam.")

    def load_sam_model(self, sam_checkpoint):
        model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
        sam_checkpoint = os.path.join(self.sam_model_dir, sam_checkpoint)
        torch.load = unsafe_torch_load
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam.eval()
        torch.load = load
        return sam

    def operation(self, *args, **kwargs):
        try:
            _batch_size = int(_batch_size)
            shared.state.interrupted = False

            output_height = 1024
            output_width = 512

            _sam_model_name = sam_model_list[0]
            _dino_model_name = dino_model_list[1]
            # _input_part_prompt = [['upper cloth'], ['pants', 'skirts'], ['shoes']]
            # _dino_text_prompt = ' . '.join([y for x in _cloth_part for y in _input_part_prompt[x]])
            # _dino_text_prompt = 'dress'
            _dino_text_prompt = 'clothing . pants . shorts . t-shirt . dress'
            _box_threshold = 0.3

            if _input_image is None:
                return None, None
            else:
                _input_image.save(f'tmp/origin_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                                  format='PNG')

                try:
                    # real people
                    if _model_mode == 0:
                        person_boxes, _ = dino_predict_internal(_input_image, _dino_model_name, "person",
                                                                _box_threshold)
                        _input_image = configure_image(_input_image, person_boxes[0],
                                                       target_ratio=output_width / output_height)
                        # _input_image = configure_image(_input_image, person_boxes[0], target_ratio=output_width / output_height)
                        pass

                    # artificial model
                    else:
                        _input_image_width, _input_image_height = _input_image.size
                        person_boxes, _ = dino_predict_internal(_input_image, _dino_model_name, "clothing",
                                                                _box_threshold)
                        person0_box = [int(x) for x in person_boxes[0]]
                        person0_width = person0_box[2] - person0_box[0]
                        person0_height = person0_box[3] - person0_box[1]
                        constant_bottom = 30
                        constant_top = 15
                        factor_bottom = 7
                        factor_top = 4
                        left_ratio = 0.1
                        right_ratio = 0.1
                        # top_ratio = 0.32
                        top_ratio = min(0.32, math.pow(person0_width / person0_height, factor_top) * constant_top)
                        bottom_ratio = min(0.54,
                                           math.pow(person0_width / person0_height, factor_bottom) * constant_bottom)
                        print(f"bottom_ratio: {bottom_ratio}")
                        print(f"top_ratio: {top_ratio}")
                        print(f"boxes: {person_boxes}")
                        print(f"width: {person0_width}")
                        print(f"height: {person0_height}")
                        print(f"increase: {person0_height * bottom_ratio}")

                        padding_left = int(person0_width * left_ratio - int(person0_box[0])) if (int(
                            person0_box[0]) / person0_width) < left_ratio else 0
                        padding_right = int(
                            person0_width * right_ratio - (_input_image_width - int(person0_box[2]))) if ((
                                                                                                                      _input_image_width - int(
                                                                                                                  person0_box[
                                                                                                                      2])) / person0_width) < right_ratio else 0
                        padding_top = int(person0_height * top_ratio - int(person0_box[1])) if (int(
                            person0_box[1]) / person0_height) < top_ratio else 0
                        padding_bottom = int(
                            person0_height * bottom_ratio - (_input_image_height - int(person0_box[3]))) if ((
                                                                                                                         _input_image_height - int(
                                                                                                                     person0_box[
                                                                                                                         3])) / person0_height) < bottom_ratio else 0

                        _input_image = padding_rgba_image_pil_to_cv(_input_image, padding_left, padding_right,
                                                                    padding_top, padding_bottom)
                        # _input_image = configure_image(_input_image, [0, 0, padding_left + _input_image_width + padding_right,
                        #                                               padding_top + _input_image_height + padding_bottom],
                        #                                target_ratio=output_width / output_height)
                        _input_image = configure_image(_input_image,
                                                       [0 if padding_left > 0 else person0_box[0] - int(
                                                           person0_width * left_ratio),
                                                        0 if padding_top > 0 else person0_box[1] - int(
                                                            person0_height * top_ratio),
                                                        padding_left + _input_image_width + padding_right if padding_right > 0 else padding_left +
                                                                                                                                    person0_box[
                                                                                                                                        2] + int(
                                                            person0_width * right_ratio),
                                                        padding_top + _input_image_height + padding_bottom if padding_bottom > 0 else padding_top +
                                                                                                                                      person0_box[
                                                                                                                                          3] + int(
                                                            person0_height * bottom_ratio)],
                                                       target_ratio=output_width / output_height)

                except Exception:
                    print(traceback.format_exc())
                    print('preprocess img error')

                if cmd_opts.debug_mode:
                    _input_image.save(f'tmp/resized_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
                                      format='PNG')

            sam_result_tmp_png_fp = []

            sam_result_gallery, sam_result = sam_predict(_dino_model_name, _dino_text_prompt,
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
            # adetail
            adetail_enabled = not cmd_opts.disable_adetailer
            face_args = {'ad_model': 'face_yolov8m.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
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
            hand_args = {'ad_model': 'hand_yolov8s.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3,
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
                        64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, None, None, False,
                        None, None,
                        False, None, None, False, 50
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
        except Exception:
            logging(
                f"[ocr predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")
        return res[0], res[0], gr.Radio.update(
            choices=[str(x) for x in range(1 if len(res[0]) == 1 else len(res[0]) - 1)], value=0), gr.Button.update(
            interactive=True), 'done.'

    def show_masks(self, image_np, masks: np.ndarray, alpha=0.5):
        image = copy.deepcopy(image_np)
        np.random.seed(0)
        for mask in masks:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
        return image.astype(np.uint8)

    def create_mask_output(self, image_np, masks, boxes_filt):
        print("Creating output image")
        mask_images, masks_gallery, matted_images = [], [], []
        boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
        for mask in masks:
            masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
            blended_image = self.show_masks(show_boxes(image_np, boxes_filt), mask)
            mask_images.append(Image.fromarray(blended_image))
            image_np_copy = copy.deepcopy(image_np)
            image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
            matted_images.append(Image.fromarray(image_np_copy))
        return mask_images + masks_gallery + matted_images

    def sam_predict(self, dino_model_name, text_prompt, box_threshold, input_image):
        positive_points = []
        negative_points = []
        print("Start SAM Processing")
        if input_image is None:
            return [], "SAM requires an input image. Please upload an image first."
        image_np = np.array(input_image)
        image_np_rgb = image_np[..., :3]
        dino_enabled = True
        boxes_filt = None
        sam_predict_result = " done."
        if dino_enabled:
            boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
            if not install_success:
                if len(positive_points) == 0 and len(negative_points) == 0:
                    return [], f"GroundingDINO installment has failed. Check your terminal for more detail and {dino_install_issue_text}. "
                else:
                    sam_predict_result += f" However, GroundingDINO installment has failed. Your process automatically fall back to point prompt only. Check your terminal for more detail and {dino_install_issue_text}. "
        print(f"Running SAM Inference {image_np_rgb.shape}")
        predictor = SamPredictor(self.sam_model)
        predictor.set_image(image_np_rgb)
        if dino_enabled and boxes_filt.shape[0] > 1:
            sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts disgarded"
            print(sam_predict_status)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=True)
            masks = masks.permute(1, 0, 2, 3).cpu().numpy()

        else:
            num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
            num_points = len(positive_points) + len(negative_points)
            if num_box == 0 and num_points == 0:
                self.garbage_collect()
                if dino_enabled and num_box == 0:
                    return [], "It seems that you are using a high box threshold with no point prompts. Please lower your box threshold and re-try."
                return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
            sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts"
            print(sam_predict_status)
            point_coords = np.array(positive_points + negative_points)
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
            box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
            masks, _, _ = predictor.predict(
                point_coords=point_coords if len(point_coords) > 0 else None,
                point_labels=point_labels if len(point_coords) > 0 else None,
                box=box,
                multimask_output=True)
            masks = masks[:, None, ...]

        # 连同区域数量最少
        masks = [masks[np.argmin([label(m)[1] for m in masks])]]
        # 最大面积
        # if len(masks) > 1:
        #     masks = [masks[np.argmax([np.count_nonzero(m) for m in masks])]]

        self.garbage_collect()
        return self.create_mask_output(image_np, masks, boxes_filt), sam_predict_status + sam_predict_result