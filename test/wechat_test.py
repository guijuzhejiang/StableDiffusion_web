# coding=utf-8
# @Time : 2023/8/24 下午2:32
# @File : wechat_test.py
import logging
import random
import string
import time
from datetime import datetime, timedelta

import pytz
import ujson
from wechatpayv3 import WeChatPay, WeChatPayType

from utils.global_vars import CONFIG

with open(CONFIG['wechatpay']['PRIVATE_KEY'], 'r') as f:
    PRIVATE_KEY = f.read()

wxpay = WeChatPay(
            wechatpay_type=WeChatPayType.NATIVE,
            mchid=CONFIG['wechatpay']['MCHID'],
            private_key=PRIVATE_KEY,
            cert_serial_no=CONFIG['wechatpay']['CERT_SERIAL_NO'],
            apiv3_key=CONFIG['wechatpay']['APIV3_KEY'],
            appid=CONFIG['wechatpay']['APPID'],
            notify_url=CONFIG['wechatpay']['NOTIFY_URL'],
            cert_dir=CONFIG['wechatpay']['CERT_DIR'],
            logger=logging.getLogger("wxpay"),
            partner_mode=CONFIG['wechatpay']['PARTNER_MODE'],
            proxy=None)


# 分为单位
amount = 1
user_id = 1
# out_trade_no=''.join([random.choice(string.ascii_letters) for c in range(8)])
out_trade_no='20231206114134150930_Ceazvtkx'
# shanghai_tz = pytz.timezone('Asia/Shanghai')
# now = datetime.now(shanghai_tz)
# now += timedelta(seconds=20)
# time_str = now.strftime('%Y-%m-%dT%H:%M:%S%z')
# time_str = time_str[:-2] + ':' + time_str[-2:]
# print(time_str)
# print(out_trade_no)
# code, message = wxpay.pay(
#     description='test',
#     out_trade_no=out_trade_no,
#     amount={'total': amount},
#     pay_type=WeChatPayType.NATIVE,
#     time_expire=time_str
# )
# time.sleep(2)
# msg = ujson.loads(message)
# print(msg)
# print(code)

# out_trade_no = '20230824092319754845_LQyiOiO'
code, message = wxpay.query(
    # transaction_id='demo-transation-id'
    out_trade_no=out_trade_no
)

msg = ujson.loads(message)
print(msg)
print(code)
