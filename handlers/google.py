# coding=utf-8
# @Time : 2023/12/18 上午9:45
# @File : line.py
import traceback

import httpx
import aiofile
import os
from sanic.response import json as sanic_json
from sanic.views import HTTPMethodView

from lib.common.common_util import encrypt
from lib.sanic_util.sanic_jinja2 import SanicJinja2
from google.oauth2 import id_token
from google.auth.transport import requests
from requests import Session

from utils.global_vars import CONFIG


class GoogleLogin(HTTPMethodView):
    """
        line登录
    """
    proxy_url = "http://127.0.0.1:7890"
    # proxy_url = "http://127.0.0.1:1095"

    # async def get(self, request):
    #     return await SanicJinja2.template_render_async("google_logging_in.html")

    async def post(self, request):
        try:
            google_login_jwt = request.form['gtoken'][0]

            req_session = Session()
            req_session.proxies = {"http": self.proxy_url, "https": self.proxy_url}
            google_info = id_token.verify_oauth2_token(google_login_jwt, requests.Request(req_session), CONFIG['googlelogin']['client_id'] if not CONFIG['local'] else "714423616983-8l6ttvp7f7nhsqg4t5q4k56onj2m2pf6.apps.googleusercontent.com")

            # create supabase account
            email = f"{google_info['sub']}@google.com".lower()
            password = encrypt(str({google_info['sub']}).lower())

            account_info = (await request.app.ctx.supabase_client.atable("account").select("id,balance,locale,nick_name").eq("google_id", str(
                google_info['sub']).lower()).execute()).data

            # 如果没有查询到则注册
            if len(account_info) == 0:
                try:
                    supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email=email,
                                                                                            password=password)
                    user_id = str(supabase_res.user.id)

                    if 'picture' in google_info.keys() and len(google_info['picture']) > 0:
                        async with httpx.AsyncClient(
                                proxies={"http://": self.proxy_url, "https://": self.proxy_url}) as client:

                            avatar_response = await client.get(google_info['picture'])
                            async with aiofile.async_open(
                                    os.path.join(CONFIG['storage_dirpath']['user_account_avatar_dir'],
                                                 f"{supabase_res.user.id}.jpg"), 'wb') as file:
                                await file.write(avatar_response.content)
                except Exception:
                    print(str(traceback.format_exc()))
                    return sanic_json({'success': False, 'message': "backend.api.error.register"})
                else:
                    res = (await request.app.ctx.supabase_client.atable("account").update(
                        {"google_id": str(google_info['sub']).lower(), 'nick_name': f'user{user_id[:8]}', "locale": ''}).eq(
                        "id", user_id).execute()).data
                    # res = (await request.app.ctx.supabase_client.atable("account").update(
                    #     {"locale": 'jp'}).eq("id", str(supabase_res.user.id)).execute()).data
            else:
                account_info = (await request.app.ctx.supabase_client.atable("account").select(
                    "id,balance,locale,nick_name").eq("google_id", str(google_info['sub']).lower()).execute()).data

            # 成功返回
            return sanic_json({'success': True,
                               'user': {'name': account_info[0]['nick_name'] if len(
                                   account_info[0]['nick_name']) > 0 else f'user{account_info[0]["id"][:8]}',
                                        'id': account_info[0]['id'],
                                        'balance': account_info[0]['balance'],
                                        'locale': account_info[0]['locale'],
                                        },
                               'expires_in': 3600
                               })

        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False, 'message': "backend.api.error.login"})
