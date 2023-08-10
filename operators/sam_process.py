# -*- encoding: utf-8 -*-
'''
@File    :   sam_process.py
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/8/10 上午10:21   zzg      1.0         None
'''

import datetime
import os
from lib.common.common_util import logging
from lib.celery_workshop.operator import Operator
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
from guiju.segment_anything_util.dino import dino_model_list, dino_predict_internal, show_boxes, dino_install_issue_text
from guiju.segment_anything_util.sam import sam_model_list, sam_predict
from modules import shared, scripts
import modules.img2img
from modules.shared import cmd_opts
import random
import string
import traceback
import cv2
import io
import math


class OperatorSAM(Operator):
    num = 2
    cache = True
    cuda = True

    def __init__(self):
        Operator.__init__(self)
        """ load sam model """
        self.sam_model_cache = OrderedDict()
        self.sam_model_dir = 'extensions/sd-webui-segment-anything/models/sam'
        self.sam_model_list = [f for f in os.listdir(self.sam_model_dir) if
                               os.path.isfile(os.path.join(self.sam_model_dir, f)) and f.split('.')[-1] != 'txt']
        self.sam_model = self.init_sam_model(self.sam_model_list[0])
        self._dino_model_name = dino_model_list[1]
        self._dino_text_prompt = 'clothing . pants . shorts . t-shirt . dress'
        self._box_threshold = 0.3
        print('SAM model is Initialized')

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
            sam_result_gallery, _ = sam_predict(self._dino_model_name, self._dino_text_prompt, self._box_threshold,
                                                kwargs['_input_image'])
        except Exception:
            logging(
                f"[SAM predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")
        return sam_result_gallery