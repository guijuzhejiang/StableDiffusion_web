# coding=utf-8
# @Time : 2023/5/16 下午6:32
# @File : sam.py
import copy
import gc
import os
from collections import OrderedDict
from PIL import Image
import numpy as np
import gradio as gr
import torch
from scipy.ndimage import label
from segment_anything import SamPredictor, sam_model_registry

from guiju.segment_anything_util.dino import show_boxes, dino_predict_internal, dino_install_issue_text
from modules.devices import torch_gc, device
from modules.safe import unsafe_torch_load, load

sam = None

refresh_symbol = '\U0001f504'       # 🔄
sam_model_cache = OrderedDict()
scripts_sam_model_dir = 'extensions/sd-webui-segment-anything/models/sam'
sd_sam_model_dir = 'extensions/sd-webui-segment-anything/models/sam'
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir
sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']

txt2img_width: gr.Slider = None
txt2img_height: gr.Slider = None
img2img_width: gr.Slider = None
img2img_height: gr.Slider = None


def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()


def garbage_collect(sam):
    gc.collect()
    torch_gc()


def init_sam_model(sam_model_name):
    print("Initializing SAM")
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        return sam
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        Exception(
            f"{sam_model_name} not found, please download model to models/sam.")


def load_sam_model(sam_checkpoint):
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    torch.load = load
    return sam


def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)


def create_mask_output(image_np, masks, boxes_filt):
    print("Creating output image")
    mask_images, masks_gallery, matted_images = [], [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        mask_images.append(Image.fromarray(blended_image))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        matted_images.append(Image.fromarray(image_np_copy))
    return mask_images + masks_gallery + matted_images


def sam_predict(dino_model_name, text_prompt, box_threshold, input_image):
    positive_points = []
    negative_points = []
    dino_preview_boxes_selection = []
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
        # valid_indices = [int(i) for i in dino_preview_boxes_selection if int(i) < boxes_filt.shape[0]]
        # boxes_filt = boxes_filt[valid_indices]
        if not install_success:
            if len(positive_points) == 0 and len(negative_points) == 0:
                return [], f"GroundingDINO installment has failed. Check your terminal for more detail and {dino_install_issue_text}. "
            else:
                sam_predict_result += f" However, GroundingDINO installment has failed. Your process automatically fall back to point prompt only. Check your terminal for more detail and {dino_install_issue_text}. "
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictor(sam)
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
            garbage_collect(sam)
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
    # masks = [masks[np.argmin([label(m)[1] for m in masks])]]
    # 最大面积
    # if len(masks) > 1:
    #     masks = [masks[np.argmax([np.count_nonzero(m) for m in masks])]]
    # first
    masks = masks[masks[0]]

    garbage_collect(sam)
    return create_mask_output(image_np, masks, boxes_filt), sam_predict_status + sam_predict_result
