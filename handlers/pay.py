# coding=utf-8
# @Time : 2023/12/29 下午4:18
# @File : pay.py
import base64
import random
import traceback
import uuid

import pytz
import string
from datetime import datetime, timedelta, date
import httpx
import ujson
from sanic.views import HTTPMethodView
from wechatpayv3 import WeChatPayType
from sanic.response import json as sanic_json

from guiju.sub import setup_cron
from lib.common.common_util import next_month_date
from utils.global_vars import CONFIG

discount_dict = {
    'first_time': [
        {
            'zh': '首次充值享8折优惠',
            'en': 'Enjoy 20% off on your first recharge',
            'ja': '初回購入時に20%オフをお楽しみください',
        },
        0.8
    ],
    'additional_notes': [
        {
            'zh': '多个折扣可叠加',
            'en': 'Multiple discounts can be stacked',
            'ja': '複数の割引を重ねて適用できる',
        },
        None
    ],
    0: [
        {
            'zh': '十二月优惠礼6折',
            'en': 'December Special Gift 60% off',
            'ja': '12月のスペシャルギフト 60%オフ',
        },
        0.6
    ],
}


class QueryDiscount(HTTPMethodView):
    """
        查询当前可用优惠
    """

    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            lang = request.args.get("lang", 'zh')

            transaction_data = (
                await request.app.ctx.supabase_client.table("transaction").select("*").eq("user_id", user_id).eq(
                    "is_plus", True).eq("status", 1).execute())
            first_charge = len(transaction_data.data) == 0

            res = []
            if first_charge:
                res.append([discount_dict['first_time'][0][lang],
                            discount_dict['first_time'][1]])

            start_date = date(2023, 12, 1)
            end_date = date(2023, 12, 29)
            if start_date <= date.today() <= end_date:
                res.append([discount_dict[0][0][lang],
                            discount_dict[0][1]])

            if len(res) > 1:
                res.append([discount_dict['additional_notes'][0][lang], 1])

            return sanic_json({'success': True, 'result': res})
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'message': 'backend.api.error.default'})


class QueryBalance(HTTPMethodView):
    """
        查余额
    """

    async def post(self, request):
        user_id = request.form['user_id'][0]
        no_confirm_rows = (
            await request.app.ctx.supabase_client.table("transaction").select("*").eq("user_id", user_id).eq("status",
                                                                                                              0).eq(
                "is_plus", True).execute()).data

        # 计算未确认支付额
        pre_charge_amount = 0
        need_del_transaction = []
        for row in no_confirm_rows:
            if '@paypal' in row['out_trade_no']:
                out_trade_no = row['out_trade_no'].replace('@paypal', '')
                url = f"{CONFIG['paypal']['base_url']}/v2/checkout/orders/{out_trade_no}/capture"

                access_token = await aspayapl_generate_ccess_token()
                async with httpx.AsyncClient(proxies={"http://": CONFIG['http_proxy'],
                                                      "https://": CONFIG['http_proxy']}) as client:
                    response = await client.post(url,
                                                 headers={"Authorization": f"Bearer {access_token}",
                                                          "Content-Type": "application/json"})

                    result = response.json()
                    if 'status' in result.keys() and result['status'] == 'COMPLETED':
                        data = (await request.app.ctx.supabase_client.table("transaction").update(
                            {"status": 1}).eq("id", row['id']).eq("is_plus", True).execute()).data
                        if len(data) == 0:
                            print(row['id'] + " update transaction false")
                        pre_charge_amount += int(row['amount'] / CONFIG['payment']['point_price']['USD'])
                    else:
                        iso_created_at = row['created_at'].split('.')[0] + '+' + row['created_at'].split('+')[-1]
                        if (datetime.now(pytz.UTC) - datetime.fromisoformat(iso_created_at).replace(
                                tzinfo=pytz.UTC)) >= timedelta(minutes=15):
                            need_del_transaction.append(row['id'])

            elif '@elepay' in row['out_trade_no']:
                code = row['out_trade_no'].replace('@elepay', '')
                url = f"https://api.elepay.io/codes/{code}"

                headers = {
                    "accept": "application/json;charset=utf-8",
                    "authorization": "Bearer sk_test_38677e6ab39d8f589894f"
                }

                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=headers)

                    result = response.json()
                    if 'status' in result.keys() and result['status'] == 'captured':
                        data = (await request.app.ctx.supabase_client.table("transaction").update(
                            {"status": 1}).eq("id", row['id']).eq("is_plus", True).execute()).data
                        if len(data) == 0:
                            print(row['id'] + " update transaction false")
                        pre_charge_amount += int(row['amount'] / CONFIG['payment']['point_price']['USD'])
                    else:
                        iso_created_at = row['created_at'].split('.')[0] + '+' + row['created_at'].split('+')[-1]
                        if (datetime.now(pytz.UTC) - datetime.fromisoformat(iso_created_at).replace(
                                tzinfo=pytz.UTC)) >= timedelta(minutes=15):
                            need_del_transaction.append(row['id'])
            else:
                code, message = request.app.ctx.wxpay.query(
                    # transaction_id='demo-transation-id'
                    out_trade_no=row['out_trade_no']
                )
                trade_message = ujson.loads(message)

                if code == 200 and trade_message['trade_state'] == 'SUCCESS':
                    data = (await request.app.ctx.supabase_client.table("transaction").update(
                        {"status": 1}).eq("id", row['id']).eq("is_plus", True).execute()).data
                    if len(data) == 0:
                        print(row['id'] + " update transaction false")
                    pre_charge_amount += int(row['amount'] / CONFIG['payment']['point_price']['RMB'])
                else:
                    iso_created_at = row['created_at'].split('.')[0] + '+' + row['created_at'].split('+')[-1]
                    if (datetime.now(pytz.UTC) - datetime.fromisoformat(iso_created_at).replace(
                            tzinfo=pytz.UTC)) >= timedelta(minutes=15):
                        need_del_transaction.append(row['id'])

        account = \
            (await request.app.ctx.supabase_client.table("account").select("*").eq("id", user_id).execute()).data[0]
        if pre_charge_amount > 0:
            data = (await request.app.ctx.supabase_client.table("account").update(
                {"balance": account['balance'] + pre_charge_amount}).eq("id", account['id']).execute()).data
        if len(need_del_transaction):
            for del_target in need_del_transaction:
                data = (await request.app.ctx.supabase_client.table("transaction").delete().eq("id",
                                                                                                del_target).execute()).data

        if hasattr(request.form, 'out_trade_no'):
            out_trade_no = request.form['out_trade_no'][0]
            code, message = request.app.ctx.wxpay.query(
                out_trade_no=out_trade_no
            )

            return sanic_json({'success': code == 200, 'balance': account['balance'] + pre_charge_amount})
        else:
            return sanic_json({'success': True, 'balance': account['balance'] + pre_charge_amount})


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""WECHAT""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


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
            (await request.app.ctx.supabase_client.table("account").select("*").eq("id", user_id).execute()).data[0]
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
            data = await request.app.ctx.supabase_client.table("transaction").insert({"user_id": user_id,
                                                                                       'amount': charge_points,
                                                                                       'is_plus': True,
                                                                                       'status': 0,
                                                                                       'out_trade_no': out_trade_no}).execute()
            qrcode_url = ujson.loads(message)['code_url']
        return sanic_json({'success': code == 200, 'qrcode_url': qrcode_url, 'out_trade_no': out_trade_no})


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""""PAYPAL""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


async def aspayapl_generate_ccess_token():
    # Encode client ID and client secret as base64
    auth = base64.b64encode(f"{CONFIG['paypal']['client_id']}:{CONFIG['paypal']['client_secret']}".encode()).decode()

    # Make a POST request to obtain the access token
    async with httpx.AsyncClient(proxies={"http://": CONFIG['http_proxy'],
                                          "https://": CONFIG['http_proxy']}) as client:
        response = await client.post(f"{CONFIG['paypal']['base_url']}/v1/oauth2/token",
                                     data="grant_type=client_credentials",
                                     headers={"Authorization": f"Basic {auth}"})
    # Parse the JSON response
    data = response.json()

    # Return the access token
    return data["access_token"]


class PayPalCreateOrder(HTTPMethodView):
    """
        请求paypal支付,创建订单
    """

    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            amount = request.form['amount'][0]
            charge_points = int(int(amount) / CONFIG['payment']['point_price']['USD'])

            account = \
                (await request.app.ctx.supabase_client.table("account").select("access_level").eq("id",
                                                                                                   user_id).execute()).data[
                    0]

            # query discount
            transaction_data = (
                await request.app.ctx.supabase_client.table("transaction").select("*").eq("user_id", user_id).eq(
                    "is_plus", True).eq("status", 1).execute())
            first_charge = len(transaction_data.data) == 0

            available_discount = []
            if first_charge:
                available_discount.append(discount_dict['first_time'][1])

            start_date = date(2023, 12, 1)
            end_date = date(2023, 12, 29)
            if start_date <= date.today() <= end_date:
                available_discount.append(discount_dict[0][1])

            fee = int(amount)
            for x in available_discount:
                fee = fee * x

            access_token = await aspayapl_generate_ccess_token()

            url = f"{CONFIG['paypal']['base_url']}/v2/checkout/orders"
            payload = {
                'intent': "CAPTURE",
                'purchase_units': [
                    {
                        'amount': {
                            'currency_code': "USD",
                            'value': f"{str(round(float(fee), 2))}" if account['access_level'] != 0 else '0.01',
                        },
                    },
                ],
            }
            async with httpx.AsyncClient(proxies={"http://": CONFIG['http_proxy'],
                                                  "https://": CONFIG['http_proxy']}) as client:
                response = await client.post(url,
                                             data=ujson.dumps(payload),
                                             headers={"Authorization": f"Bearer {access_token}",
                                                      "Content-Type": "application/json"})

                if 'id' in response.json().keys():
                    data = await request.app.ctx.supabase_client.table("transaction").insert({"user_id": user_id,
                                                                                               'amount': charge_points,
                                                                                               'is_plus': True,
                                                                                               'status': 0,
                                                                                               'out_trade_no': f"{response.json()['id']}@paypal"}).execute()

                return sanic_json(response.json(), status=response.status_code)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False}, status=500)


class PayPalCaptureOrder(HTTPMethodView):
    """
        请求paypal支付，查询订单
    """

    async def post(self, request):
        out_trade_no = request.form['out_trade_no'][0]

        access_token = await aspayapl_generate_ccess_token()
        url = f"{CONFIG['paypal']['base_url']}/v2/checkout/orders/{out_trade_no}/capture"

        transaction_data = (await request.app.ctx.supabase_client.table("transaction").select("*").eq("out_trade_no",
                                                                                                       f"{out_trade_no}@paypal").eq(
            "is_plus", True).execute()).data[0]

        async with httpx.AsyncClient(proxies={"http://": CONFIG['http_proxy'],
                                              "https://": CONFIG['http_proxy']}) as client:
            response = await client.post(url,
                                         headers={"Authorization": f"Bearer {access_token}",
                                                  "Content-Type": "application/json"})

            result = response.json()
            if result['status'] == 'COMPLETED':
                data = (await request.app.ctx.supabase_client.table("transaction").update(
                    {"status": 1}).eq("out_trade_no", f"{out_trade_no}@paypal").eq("is_plus", True).execute()).data

                account = \
                    (await request.app.ctx.supabase_client.table("account").select("balance").eq("id",
                                                                                                  transaction_data[
                                                                                                      'user_id']).execute()).data[
                        0]
                data = (await request.app.ctx.supabase_client.table("account").update(
                    {"balance": account['balance'] + transaction_data['amount']}).eq("id", transaction_data[
                    'user_id']).execute()).data

            return sanic_json(response.json(), status=response.status_code)


class PayPalCreateSub(HTTPMethodView):
    """
        请求paypal支付,创建訂閱
    """

    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            subscription_id = request.form['subscription_id'][0]
            vip_level = int(request.form['vip_level'][0])

            if vip_level == 1:
                add_balance = 220
            elif vip_level == 2:
                add_balance = 720
            # else:
            #     add_balance = 1625

            account = (await request.app.ctx.supabase_client.table("account").select("balance").eq("id",
                                                                                                    user_id).execute()).data[
                0]
            subscription = (await request.app.ctx.supabase_client.table("subscription").select("*").eq("user_id",
                                                                                                        user_id).execute()).data

            if len(subscription) > 0:
                data = await request.app.ctx.supabase_client.table("subscription").update(
                    {'subscription_id': subscription_id,
                     'supplier': 'paypal',
                     }).eq("user_id", user_id).execute()

            else:
                data = await request.app.ctx.supabase_client.table("subscription").insert({"user_id": user_id,
                                                                                            'subscription_id': subscription_id,
                                                                                            'supplier': 'paypal',
                                                                                            }).execute()

            data = await request.app.ctx.supabase_client.table("transaction").insert({"user_id": user_id,
                                                                                       'out_trade_no': f'@subpaypal_{subscription_id}',
                                                                                       'amount': add_balance,
                                                                                       'is_plus': True,
                                                                                       'status': 1,
                                                                                       }).execute()
            res = (await request.app.ctx.supabase_client.table("account").update(
                {"balance": account['balance'] + add_balance, 'vip_level': vip_level}).eq("id", user_id).execute()).data

            # 获取次月结算日
            current_date = datetime.now()
            next_checkout_date = next_month_date(src_date=current_date) + timedelta(days=1)
            setup_cron(subscription_id, next_checkout_date)

            return sanic_json({'success': True}, status=200)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False}, status=500)


class CheckVip(HTTPMethodView):
    """
        检查是不是vip
    """

    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            res = (await request.app.ctx.supabase_client.table("account").select("vip_level").eq("id",
                                                                                                  user_id).execute()).data

            return sanic_json({'success': res[0]['vip_level'] > 0}, status=200)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False}, status=500)


class PayPalCancelSub(HTTPMethodView):
    """
        取消订阅
    """

    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            subscription = (await request.app.ctx.supabase_client.table("subscription").select("*").eq("user_id",
                                                                                                        user_id).execute()).data
            if len(subscription) > 0:

                # get access token
                access_token = await aspayapl_generate_ccess_token()
                # Create product
                url = f"{CONFIG['paypal']['base_url']}/v1/billing/subscriptions/{subscription[0]['subscription_id']}/cancel"
                payload = {
                    "reason": "Not satisfied with the service"
                }

                async with httpx.AsyncClient(proxies={"http://": CONFIG['http_proxy'],
                                                      "https://": CONFIG['http_proxy']}) as client:
                    response = await client.post(url,
                                                 data=ujson.dumps(payload),
                                                 headers={"Authorization": f"Bearer {access_token}",
                                                          "Content-Type": "application/json"})
                    # print(response)
                    # response_json = response.json()
                    # print(response_json)

                res = (await request.app.ctx.supabase_client.table("account").update({"vip_level": 0}).eq("id",
                                                                                                           user_id).execute()).data
                data = (await request.app.ctx.supabase_client.table("subscription").delete().eq("user_id",
                                                                                                 user_id).execute()).data

                return sanic_json({'success': True})
            else:
                return sanic_json({'success': False, 'result': 'backend.api.error.no-subscription'})

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.default'})


class PayPalWebhook(HTTPMethodView):
    """
        paypal webhook
    """

    async def post(self, request):
        try:
            print(request.json)
            print(request.json['event_type'])
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.default'})


class ElepayCreateEasyQR(HTTPMethodView):
    """
        創建EasyQR
    """

    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            amount = request.form['amount'][0]
            charge_points = int(int(amount) / CONFIG['payment']['point_price']['USD'])

            account = \
                (await request.app.ctx.supabase_client.table("account").select("access_level").eq("id",
                                                                                                   user_id).execute()).data[
                    0]

            # query discount
            transaction_data = (
                await request.app.ctx.supabase_client.table("transaction").select("*").eq("user_id", user_id).eq(
                    "is_plus", True).eq("status", 1).execute())
            first_charge = len(transaction_data.data) == 0

            available_discount = []
            if first_charge:
                available_discount.append(discount_dict['first_time'][1])

            start_date = date(2023, 12, 1)
            end_date = date(2023, 12, 29)
            if start_date <= date.today() <= end_date:
                available_discount.append(discount_dict[0][1])

            fee = int(amount)
            for x in available_discount:
                fee = fee * x

            access_token = "sk_test_38677e6ab39d8f589894f"

            url = "https://api.elepay.io/codes"

            headers = {
                "accept": "application/json;charset=utf-8",
                "content-type": "application/json;charset=utf-8",
                "authorization": f"Bearer {access_token}"
            }
            payload = {
                "currency": "JPY",
                # 'value': f"{str(round(float(fee), 2))}" if account['access_level'] != 0 else '0.01',
                "amount": int(fee*157.05) if account['access_level'] != 0 else 1,
                "orderNo": str(uuid.uuid4()),
                "frontUrl": "http://localhost/elepay/done" if CONFIG['debug_mode'] else CONFIG['server']['client_access_url'].replace('service', 'elepay/done'),
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(url,
                                             data=ujson.dumps(payload),
                                             headers=headers)

                if response.status_code == 201:

                    if 'id' in response.json().keys():
                        data = await request.app.ctx.supabase_client.table("transaction").insert({"user_id": user_id,
                                                                                               'amount': charge_points,
                                                                                               'is_plus': True,
                                                                                               'status': 0,
                                                                                               'out_trade_no': f"{response.json()['id']}@elepay"}).execute()

                    return sanic_json({'success': True, 'result': response.json()}, status=response.status_code)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False}, status=500)


class ElepayQueryPayment(HTTPMethodView):
    """
        查询微信支付是否成功
    """

    async def post(self, request):
        code = request.form['out_trade_no'][0].replace('@elepay', '')
        url = f"https://api.elepay.io/codes/{code}"

        headers = {
            "accept": "application/json;charset=utf-8",
            "authorization": "Bearer sk_test_38677e6ab39d8f589894f"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url,
                                         headers=headers)

        return sanic_json({'success': response.status_code == 200 and response.json()["status"] == "captured", 'result': response.json()})

