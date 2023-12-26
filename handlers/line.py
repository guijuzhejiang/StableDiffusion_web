# coding=utf-8
# @Time : 2023/12/18 上午9:45
# @File : line.py
import traceback
import aiofile
import os

from sanic.response import json as sanic_json

import httpx
from sanic.views import HTTPMethodView
from gotrue import check_response

from lib.common.common_util import encrypt
from lib.sanic_util.sanic_jinja2 import SanicJinja2
from utils.global_vars import CONFIG


class LineLogin(HTTPMethodView):
    """
        line登录
    """
    proxy_url = "http://127.0.0.1:1095"

    async def get(self, request):
        return await SanicJinja2.template_render_async("line_logging_in.html")

    async def post(self, request):
        try:
            state = request.form['state'][0]
            code = request.form['code'][0]
            print(code)
            # 发起GET请求
            async with httpx.AsyncClient(proxies={"http://": self.proxy_url, "https://": self.proxy_url}) as client:
                # req token
                token_form_data = {
                    "grant_type": "authorization_code",
                    "code": code,
                    "state": state,
                    "redirect_uri": CONFIG['linelogin']['redirect_uri'],
                    "client_id": CONFIG['linelogin']['client_id'],
                    "client_secret": CONFIG['linelogin']['client_secret'],
                }

                # Send POST request with application/x-www-form-urlencoded content type
                token_response = await client.post(CONFIG['linelogin']['token_url'], data=token_form_data,
                                      headers={"Content-Type": "application/x-www-form-urlencoded"})

                if token_response.status_code != 200:
                    return sanic_json({'success': False, 'message': "backend.api.error.register"})


                # {
                #     "access_token": "bNl4YEFPI/hjFWhTqexp4MuEw5YPs...",
                #     "expires_in": 2592000,
                #     "id_token": "eyJhbGciOiJIUzI1NiJ9...",
                #     "refresh_token": "Aa1FdeggRhTnPNNpxr8p",
                #     "scope": "profile",
                #     "token_type": "Bearer"
                # }
                token_res = token_response.json()

                # req line account info
                line_form_data = {
                    "client_id": CONFIG['linelogin']['client_id'],
                    "id_token": token_res['id_token'],
                }

                # Send POST request with application/x-www-form-urlencoded content type
                line_response = await client.post(CONFIG['linelogin']['verify_url'], data=line_form_data,
                                                   headers={"Content-Type": "application/x-www-form-urlencoded"})

                if line_response.status_code != 200:
                    return sanic_json({'success': False, 'message': "backend.api.error.register"})
                # {
                #     "iss": "https://access.line.me",
                #     "sub": "U1234567890abcdef1234567890abcdef",
                #     "aud": "1234567890",
                #     "exp": 1504169092,
                #     "iat": 1504263657,
                #     "nonce": "0987654asdf",
                #     "amr": ["pwd"],
                #     "name": "Taro Line",
                #     "picture": "https://sample_line.me/aBcdefg123456",
                #     "email": "taro.line@example.com"
                # }
                line_info = line_response.json()

                email = f"{line_info['sub']}@line.com".lower()
                password = encrypt(str({line_info['sub']}).lower())
                # result_user = {'username': email,
                #                'password': password,
                #                'avatar': line_info['picture'] if 'picture' in line_info.keys() else '',
                #                'name': line_info['name'],
                #                }

                # supabase 检查有没有改用户，没有就注册
                # users = await request.app.ctx.supabase_client.auth.async_list_users()
                # h = request.app.ctx.supabase_client.auth.headers
                # response = await request.app.ctx.supabase_client.auth.async_api.http_client.get(
                #     f"{request.app.ctx.supabase_client.auth.url}/admin/users?per_page=9999", headers=h)
                # check_response(response)
                # users = response.json().get("users")
                # # if users is None:
                # #     return sanic_json({'success': False, 'message': "登录失败"})
                # if not isinstance(users, list):
                #     return sanic_json({'success': False, 'message': "backend.api.error.default"})

                # users_email = [u['email'] for u in users]

                id_res = (await request.app.ctx.supabase_client.atable("account").select("id").eq("line_id", str(
                    line_info['sub']).lower()).execute()).data
                # 如果没有查询到则注册
                if len(id_res) == 0:
                    try:
                        supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email=email,
                                                                                                password=password)
                        user_id = supabase_res.user.id

                        if 'picture' in line_info.keys():
                            avatar_response = await client.get(line_info['picture'])
                            async with aiofile.async_open(
                                    os.path.join(CONFIG['storage_dirpath']['user_account_avatar_dir'],
                                                 f"{supabase_res.user.id}.jpg"), 'wb') as file:
                                await file.write(avatar_response.body)
                    except Exception:
                        print(str(traceback.format_exc()))
                        return sanic_json({'success': False, 'message': "backend.api.error.register"})
                    else:
                        res = (await request.app.ctx.supabase_client.atable("account").update(
                            {"line_id": str(line_info['sub']).lower(), 'nick_name': line_info['name'], "locale": 'jp'}).eq(
                            "id", user_id).execute()).data
                        # res = (await request.app.ctx.supabase_client.atable("account").update(
                        #     {"locale": 'jp'}).eq("id", str(supabase_res.user.id)).execute()).data
                else:
                    user_id = id_res[0]['id']

                account_info = (await request.app.ctx.supabase_client.atable("account").select(
                    "id,balance,locale,nick_name").eq("line_id", str(line_info['sub']).lower()).execute()).data

                # 成功返回
                return sanic_json({'success': True,
                                   'user': {'username': account_info[0]['nick_name'],
                                            'id': account_info[0]['id'],
                                            'balance': account_info[0]['balance'],
                                            'locale': account_info[0]['locale'],
                                            },
                                   'expires_in': 3600
                                   })
        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False, 'message': "backend.api.error.default"})