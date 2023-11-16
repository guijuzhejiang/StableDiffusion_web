import hashlib
import hmac
import os
import random
import base64
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


def pil_to_base64(pil_image, to_html=True):
    buff = BytesIO()
    pil_image.save(buff, format="JPEG")
    base64_string = base64.b64encode(buff.getvalue()).decode('utf-8')

    if to_html:
        base64_string = 'data:image/jpeg;base64,' + base64_string
    return base64_string


if __name__ == '__main__':
    print(encrypt("data"))