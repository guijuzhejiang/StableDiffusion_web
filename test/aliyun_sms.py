# coding=utf-8
# @Time : 2023/12/27 上午10:39
# @File : aliyun_sms.py
import traceback

import ujson
from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
import os

from utils.global_vars import CONFIG

os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = CONFIG['aliyun']['sms']['access_key_id']
os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = CONFIG['aliyun']['sms']['access_key_secret']

try:
    exp_secs = 300
    phone = '+8613590742818'
    captcha = '123456'
    config = open_api_models.Config(
        # 必填，您的 AccessKey ID,
        access_key_id=os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'],
        # 必填，您的 AccessKey Secret,
        access_key_secret=os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
    )
    # Endpoint 请参考 https://api.aliyun.com/product/Dysmsapi
    config.endpoint = f'dysmsapi.aliyuncs.com'
    client = Dysmsapi20170525Client(config)
    send_sms_request = dysmsapi_20170525_models.SendSmsRequest(
        phone_numbers=phone,
        sign_name='幻景AI',
        template_code=CONFIG['aliyun']['sms']['global_template_code'],
        template_param=ujson.dumps({'code': captcha}),
    )
    try:
        # 复制代码运行请自行打印 API 的返回值
        res = client.send_sms_with_options(send_sms_request, util_models.RuntimeOptions())
    except Exception as error:
        print(error)
    else:
        print(res.body.code)
except Exception:
    print(traceback.format_exc())