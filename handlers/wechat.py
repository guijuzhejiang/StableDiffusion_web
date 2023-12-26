# coding=utf-8
# @Time : 2023/12/15 下午2:21
# @File : wechat.py
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


class ReqPayQRCode(HTTPMethodView):
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


class QueryPayment(HTTPMethodView):
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


class WeChatLogin(HTTPMethodView):
    """
        微信登录
    """

    async def get(self, request):
        # state = request.args.get('state')
        # code = request.args.get('code')
        return await SanicJinja2.template_render_async("loggingin.html")

    async def post(self, request):
        try:
            state = request.form['state'][0]
            code = request.form['code'][0]
            print(code)
            # 发起GET请求
            async with httpx.AsyncClient() as client:
                # 替换下面的URL为你要发送GET请求的目标URL
                response = await client.get(
                    f'https://api.weixin.qq.com/sns/oauth2/access_token?appid={CONFIG["wechatlogin"]["appid"]}&secret={CONFIG["wechatlogin"]["secret"]}&code={code}&grant_type=authorization_code')
                # print(
                #     f'https://api.weixin.qq.com/sns/oauth2/access_token?appid={CONFIG["wechatlogin"]["appid"]}&secret={CONFIG["wechatlogin"]["secret"]}&code={code}&grant_type=authorization_code')

                # 检查响应状态码
                if response.status_code == 200:
                    # 获取响应的 JSON 数据
                    wechat_data = response.json()
                    print(wechat_data)
                    email = f"{wechat_data['openid']}@wechat.com".lower()
                    # result_user = {'username': email, 'password': password}
                    id_res = (await request.app.ctx.supabase_client.atable("account").select("id").eq("wechat_id", str(
                        wechat_data['openid'].lower())).execute()).data

                    # supabase 检查有没有改用户，没有就注册
                    # users = await request.app.ctx.supabase_client.auth.async_list_users()
                    # h = request.app.ctx.supabase_client.auth.headers
                    # response = await request.app.ctx.supabase_client.auth.async_api.http_client.get(
                    #     f"{request.app.ctx.supabase_client.auth.url}/admin/users?per_page=9999", headers=h)
                    # check_response(response)
                    # users = response.json().get("users")
                    # if users is None:
                    #     return sanic_json({'success': False, 'message': "登录失败"})
                    # if not isinstance(users, list):
                    #     return sanic_json({'success': False, 'message': "backend.api.error.default"})

                    # users_email = [u['email'] for u in users]

                    # 如果没有查询到则注册
                    if len(id_res) == 0:
                        password = encrypt(str({wechat_data['openid']}).lower())

                        try:
                            supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email=email,
                                                                                                    password=password)

                            user_id = supabase_res.user.id
                            # save avatar
                            try:
                                # 获取微信头像
                                response_uinfo = await client.get(
                                    f'https://api.weixin.qq.com/sns/userinfo?access_token={wechat_data["access_token"]}&openid={wechat_data["openid"]}')
                                if response_uinfo.status_code == 200:
                                    wechat_uinfo = response_uinfo.json()

                                avatar_response = await client.get(wechat_uinfo['headimgurl'])
                                async with aiofile.async_open(
                                        os.path.join(CONFIG['storage_dirpath']['user_account_avatar_dir'],
                                                     f"{supabase_res.user.id}.jpg"), 'wb') as file:
                                    await file.write(avatar_response.body)

                            except Exception:
                                print(str(traceback.format_exc()))

                            # result_user['avatar'] = f'service/user/image/fetch?category=account_avatar&uid={str(supabase_res.user.id)}'
                            data = (await request.app.ctx.supabase_client.atable("account").update(
                                {"wechat_id": str(wechat_data['openid']).lower(), 'nick_name': wechat_uinfo['nickname']}).eq(
                                "id", user_id).execute()).data

                        except Exception:
                            print(str(traceback.format_exc()))
                            return sanic_json({'success': False, 'message': "backend.api.error.register"})
                    else:
                        user_id = id_res[0]['id']

                    account_info = (await request.app.ctx.supabase_client.atable("account").select(
                        "id,balance,locale,nick_name").eq("wechat_id", str(wechat_data['openid']).lower()).execute()).data

                    # 成功返回
                    return sanic_json({'success': True,
                                       'user': {'name': account_info[0]['nick_name'],
                                                'id': account_info[0]['id'],
                                                'balance': account_info[0]['balance'],
                                                'locale': account_info[0]['locale'],
                                                },
                                       'expires_in': 3600
                                       })
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    return sanic_json({'success': False, 'message': "backend.api.error.login"})
        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False, 'message': "backend.api.error.default"})
