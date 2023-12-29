# coding=utf-8
# @Time : 2023/12/29 下午4:18
# @File : pay.py
import os
import random
import traceback
from datetime import datetime, timedelta
import string

import httpx
import ujson
import aiofile
from sanic.views import HTTPMethodView
from wechatpayv3 import WeChatPayType
from sanic.response import json as sanic_json
from gotrue import check_response

from lib.sanic_util.sanic_jinja2 import SanicJinja2
from lib.common.common_util import encrypt
from utils.global_vars import CONFIG


class WechatQueryPayment(HTTPMethodView):
    """
        查询微信支付是否成功
    """

    async def post(self, request):
        out_trade_no = request.form['out_trade_no'][0]
        code, message = request.app.ctx.wxpay.query(
            out_trade_no=out_trade_no,
        )

        res = ujson.loads(message)

        return sanic_json({'success': code == 200, 'message': res})


class WechatReqPayQRCode(HTTPMethodView):
    """
        请求微信支付码
    """

    async def post(self, request):
        user_id = request.form['user_id'][0]
        account = \
            (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
        # 以native下单为例，下单成功后即可获取到'code_url'，将'code_url'转换为二维码，并用微信扫码即可进行支付测试。
        out_trade_no = datetime.now().strftime('%Y%m%d%H%M%S%f') + '_' + ''.join(
            [random.choice(string.ascii_letters) for c in range(8)])
        # 分为单位
        charge_points = int(request.form['amount'][0])
        charge_fee = float(request.form['fee'][0])
        total_price = int(charge_fee * 100) if account['access_level'] != 0 else 1
        description = f'收银台(充值{charge_points}点)'
        code, message = request.app.ctx.wxpay.pay(
            description=description,
            out_trade_no=out_trade_no,
            amount={'total': total_price},
            pay_type=WeChatPayType.NATIVE
        )
        qrcode_url = ''
        if code == 200:
            data = await request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                                       'amount': charge_points,
                                                                                       'is_plus': True,
                                                                                       'status': 0,
                                                                                       'out_trade_no': out_trade_no}).execute()
            qrcode_url = ujson.loads(message)['code_url']
        return sanic_json({'success': code == 200, 'qrcode_url': qrcode_url, 'out_trade_no': out_trade_no})


class PayPalCreateOrder(HTTPMethodView):
    """
        请求paypal支付
    """
    async def post(self, request):
        user_id = request.form['user_id'][0]
        account = \
            (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
        # 以native下单为例，下单成功后即可获取到'code_url'，将'code_url'转换为二维码，并用微信扫码即可进行支付测试。
        out_trade_no = datetime.now().strftime('%Y%m%d%H%M%S%f') + '_' + ''.join(
            [random.choice(string.ascii_letters) for c in range(8)])
        # 分为单位
        charge_points = int(request.form['amount'][0])
        charge_fee = float(request.form['fee'][0])
        total_price = int(charge_fee * 100) if account['access_level'] != 0 else 1
        description = f'收银台(充值{charge_points}点)'
        code, message = request.app.ctx.wxpay.pay(
            description=description,
            out_trade_no=out_trade_no,
            amount={'total': total_price},
            pay_type=WeChatPayType.NATIVE
        )
        qrcode_url = ''
        if code == 200:
            data = await request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                                       'amount': charge_points,
                                                                                       'is_plus': True,
                                                                                       'status': 0,
                                                                                       'out_trade_no': out_trade_no}).execute()
            qrcode_url = ujson.loads(message)['code_url']
        return sanic_json({'success': code == 200, 'qrcode_url': qrcode_url, 'out_trade_no': out_trade_no})


class PayPalCaptureOrder(HTTPMethodView):
    """
        请求paypal支付
    """
    async def post(self, request):
        out_trade_no = request.form['out_trade_no'][0]
        # account = \
        #     (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
        # # 以native下单为例，下单成功后即可获取到'code_url'，将'code_url'转换为二维码，并用微信扫码即可进行支付测试。
        # out_trade_no = datetime.now().strftime('%Y%m%d%H%M%S%f') + '_' + ''.join(
        #     [random.choice(string.ascii_letters) for c in range(8)])
        # # 分为单位
        # charge_points = int(request.form['amount'][0])
        # charge_fee = float(request.form['fee'][0])
        # total_price = int(charge_fee * 100) if account['access_level'] != 0 else 1
        # description = f'收银台(充值{charge_points}点)'
        # code, message = request.app.ctx.wxpay.pay(
        #     description=description,
        #     out_trade_no=out_trade_no,
        #     amount={'total': total_price},
        #     pay_type=WeChatPayType.NATIVE
        # )
        # qrcode_url = ''
        # if code == 200:
        #     data = await request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
        #                                                                                'amount': charge_points,
        #                                                                                'is_plus': True,
        #                                                                                'status': 0,
        #                                                                                'out_trade_no': out_trade_no}).execute()
        #     qrcode_url = ujson.loads(message)['code_url']
        return sanic_json({'success': True, 'details': '', 'out_trade_no': out_trade_no})

