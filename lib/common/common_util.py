import calendar
import datetime
import hashlib
import hmac
import os
import random
import base64

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# 定义密钥（应保持安全，不要硬编码在代码中）
key = 'guijutech1201!'.encode('utf-8')  # 用字节表示的密钥

# 加密函数
def encrypt(data):
    h = hmac.new(key, data.encode('utf-8'), hashlib.sha256)
    return h.hexdigest()


def logging(msg, fp, print_msg=False):
    if print_msg:
        print(msg)
    mkdir(os.path.dirname(fp))

    with open(fp, 'a') as f:
        f.write(msg + "\n")


def mkdir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)


def generate_random(num):
    alphabet = 'abcdefghijklmnopqrstuvwxyz_-ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join([random.choice(alphabet) for i in range(num)])


def generate_random_digits(num=6):
    return ''.join([str(n) for n in random.sample(range(10), num)])


def base64_to_pil(base64_string, from_html=True):
    if from_html:
        base64_string = base64_string.split(',')[1]
    # 将 Base64 编码的字符串解码为字节数据
    image_data = base64.b64decode(base64_string)

    # 将字节数据转换为 PIL 图像对象
    pil_image = Image.open(BytesIO(image_data))

    return pil_image


def pil2cv(pil_image):
    numpy_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


def pil_to_base64(pil_image, to_html=True):
    buff = BytesIO()
    pil_image.save(buff, format="JPEG")
    base64_string = base64.b64encode(buff.getvalue()).decode('utf-8')

    if to_html:
        base64_string = 'data:image/jpeg;base64,' + base64_string
    return base64_string


def uuid_to_number_string(uuid_str):
    # 将UUID字符串转换为整数
    uuid_int = int(uuid_str.replace('-', ''), 16)

    # 将整数转换为字符串
    return str(uuid_int)


def next_month_date(src_date=None):
    if src_date is None:
        src_date = datetime.datetime.now()

    next_m = 1 if src_date.month == 12 else src_date.month + 1
    next_y = src_date.year + 1 if src_date.month == 12 else src_date.year
    last_d_next_y = calendar.monthrange(next_y, next_m)[1]

    if src_date.day > last_d_next_y:
        return datetime.datetime(next_y, next_m, last_d_next_y, 0, 0, 0)
    else:
        return datetime.datetime(next_y, next_m, src_date.day, 0, 0, 0)


if __name__ == '__main__':
    print(encrypt("data"))