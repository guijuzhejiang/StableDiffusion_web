# -*- encoding: utf-8 -*-
'''
@File    :   ocr_process.py    
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/21 上午10:21   ray      1.0         None
'''

# import lib
import base64
import copy
import datetime
import math
import os
import re
import sys
import traceback
from importlib import import_module
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image

from lib.common.common_util import logging, point_distance_line, get_vertical_line, cross_point
from lib.redis_pipline.operator import Operator
from utils.global_vars import CONFIG


class OperatorOCR(Operator):
    num = 1
    cache = True
    cuda = True

    def __init__(self):
        Operator.__init__(self)

        """ load ocr """
        self.k = 'tt'
        print('init')


    def operation(self, *args, **kwargs):
        try:
            print('operation')
        except Exception:
            logging(
                f"[ocr predict fatal error][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

            return 'super().return_msg(msg_pack)'
