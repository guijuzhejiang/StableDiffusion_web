# coding=utf-8
# @Time : 2023/12/29 下午4:18
# @File : login.py
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
            req_session.proxies = {"http": CONFIG['http_proxy'], "https": CONFIG['http_proxy']}
            google_info = id_token.verify_oauth2_token(google_login_jwt, requests.Request(req_session), CONFIG['googlelogin']['client_id'] if not CONFIG['local'] else "714423616983-8l6ttvp7f7nhsqg4t5q4k56onj2m2pf6.apps.googleusercontent.com")

            # create supabase account
            email = f"{google_info['sub']}@google.com".lower()
            password = encrypt(str({google_info['sub']}).lower())

            account_info = (await request.app.ctx.supabase_client.table("account").select("id,balance,locale,nick_name,vip_level").eq("google_id", str(
                google_info['sub']).lower()).execute()).data

            # 如果没有查询到则注册
            if len(account_info) == 0:
                try:
                    supabase_res = await request.app.ctx.supabase_client.auth.sign_up(credentials={"email": email, "password": password})
                    user_id = str(supabase_res.user.id)

                    if 'picture' in google_info.keys() and len(google_info['picture']) > 0:
                        async with httpx.AsyncClient(
                                proxies={"http://": CONFIG['http_proxy'], "https://": CONFIG['http_proxy']}) as client:

                            avatar_response = await client.get(google_info['picture'])
                            async with aiofile.async_open(
                                    os.path.join(CONFIG['storage_dirpath']['user_account_avatar'],
                                                 f"{supabase_res.user.id}.jpg"), 'wb') as file:
                                await file.write(avatar_response.content)
                except Exception:
                    print(str(traceback.format_exc()))
                    return sanic_json({'success': False, 'message': "backend.api.error.register"})
                else:
                    res = (await request.app.ctx.supabase_client.table("account").update(
                        {"google_id": str(google_info['sub']).lower(), 'nick_name': f'user{user_id[:8]}', "locale": ''}).eq(
                        "id", user_id).execute()).data
                    # res = (await request.app.ctx.supabase_client.table("account").update(
                    #     {"locale": 'jp'}).eq("id", str(supabase_res.user.id)).execute()).data
                    account_info = (await request.app.ctx.supabase_client.table("account").select(
                        "id,balance,locale,nick_name,vip_level").eq("google_id", str(google_info['sub']).lower()).execute()).data
            else:
                account_info = (await request.app.ctx.supabase_client.table("account").select(
                    "id,balance,locale,nick_name,vip_level").eq("google_id", str(google_info['sub']).lower()).execute()).data

            # 成功返回
            return sanic_json({'success': True,
                               'user': {'name': account_info[0]['nick_name'] if len(account_info[0]['nick_name']) > 0 else f'user{account_info[0]["id"][:8]}',
                                        'id': account_info[0]['id'],
                                        'vip_level': account_info[0]['vip_level'],
                                        'balance': account_info[0]['balance'],
                                        'locale': account_info[0]['locale'],
                                        },
                               'expires_in': 3600
                               })

        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False, 'message': "backend.api.error.login"})


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
                    id_res = (await request.app.ctx.supabase_client.table("account").select("id").eq("wechat_id", str(
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
                            supabase_res = await request.app.ctx.supabase_client.auth.sign_up(credentials={"email": email, "password": password})

                            user_id = supabase_res.user.id
                            # save avatar
                            try:
                                # 获取微信头像
                                response_uinfo = await client.get(
                                    f'https://api.weixin.qq.com/sns/userinfo?access_token={wechat_data["access_token"]}&openid={wechat_data["openid"]}')
                                if response_uinfo.status_code == 200:
                                    wechat_uinfo = response_uinfo.json()
                                    data = (await request.app.ctx.supabase_client.table("account").update(
                                        {"wechat_id": str(wechat_data['openid']).lower(),
                                         'nick_name': wechat_uinfo['nickname']}).eq(
                                        "id", user_id).execute()).data
                                    if wechat_uinfo['headimgurl']:
                                        avatar_response = await client.get(wechat_uinfo['headimgurl'])
                                        async with aiofile.async_open(
                                                os.path.join(CONFIG['storage_dirpath']['user_account_avatar'],
                                                             f"{supabase_res.user.id}.jpg"), 'wb') as file:
                                            await file.write(avatar_response.content)

                                else:
                                    return sanic_json({'success': False, 'message': "backend.api.error.register"})
                            except Exception:
                                print(str(traceback.format_exc()))

                        except Exception:
                            print(str(traceback.format_exc()))
                            return sanic_json({'success': False, 'message': "backend.api.error.register"})
                    else:
                        user_id = id_res[0]['id']

                    account_info = (await request.app.ctx.supabase_client.table("account").select(
                        "id,balance,locale,nick_name,vip_level").eq("wechat_id", str(wechat_data['openid']).lower()).execute()).data

                    # 成功返回
                    return sanic_json({'success': True,
                                       'user': {'name': account_info[0]['nick_name'] if len(account_info[0]['nick_name']) > 0 else f'user{account_info[0]["id"][:8]}',
                                                'id': account_info[0]['id'],
                                                'balance': account_info[0]['balance'],
                                                'vip_level': account_info[0]['vip_level'],
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


class PasswordLogin(HTTPMethodView):
    """
        密码登录
    """
    async def post(self, request):
        try:
            phone = request.form['phone'][0]
            password = request.form['password'][0]

            # supabase_res = await request.app.ctx.supabase_client.auth.async_sign_in(email=phone, password=password)
            supabase_res = await request.app.ctx.supabase_client.auth.sign_in_with_password(credentials={'email': phone, 'password': password})
            # await request.app.ctx.supabase_client.auth.sign_out()
            # supabase_res = request.app.ctx.supabase_client_sync.auth.sign_in_with_password(credentials={'email': phone, 'password': password})
            account_info = (await request.app.ctx.supabase_client.table("account").select(
                "id,balance,locale,nick_name,vip_level").eq("id", supabase_res.user.id).execute()).data
            return sanic_json({'success': True,
                               'user': {'name': account_info[0]['nick_name'],
                                        'id': account_info[0]['id'],
                                        'balance': account_info[0]['balance'],
                                        'locale': account_info[0]['locale'],
                                        'vip_level': account_info[0]['vip_level'],
                                        },
                               'expires_in': 3600
                               })
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})


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
            async with httpx.AsyncClient(proxies={"http://": CONFIG['http_proxy'], "https://": CONFIG['http_proxy']}) as client:
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

                account_info = (await request.app.ctx.supabase_client.table("account").select("id,balance,locale,nick_name,vip_level").eq("line_id", str(
                    line_info['sub']).lower()).execute()).data
                # 如果没有查询到则注册
                if len(account_info) == 0:
                    try:
                        supabase_res = await request.app.ctx.supabase_client.auth.sign_up(credentials={"email": email, "password": password})
                        user_id = str(supabase_res.user.id)

                        if 'picture' in line_info.keys():
                            avatar_response = await client.get(line_info['picture'])
                            async with aiofile.async_open(
                                    os.path.join(CONFIG['storage_dirpath']['user_account_avatar'],
                                                 f"{supabase_res.user.id}.jpg"), 'wb') as file:
                                await file.write(avatar_response.content)
                    except Exception:
                        print(str(traceback.format_exc()))
                        return sanic_json({'success': False, 'message': "backend.api.error.register"})
                    else:
                        res = (await request.app.ctx.supabase_client.table("account").update(
                            {"line_id": str(line_info['sub']).lower(), 'nick_name': f'user{user_id[:8]}', "locale": 'jp'}).eq(
                            "id", user_id).execute()).data
                        # res = (await request.app.ctx.supabase_client.table("account").update(
                        #     {"locale": 'jp'}).eq("id", str(supabase_res.user.id)).execute()).data
                else:
                    account_info = (await request.app.ctx.supabase_client.table("account").select(
                        "id,balance,locale,nick_name,vip_level").eq("line_id", str(line_info['sub']).lower()).execute()).data

                # 成功返回
                return sanic_json({'success': True,
                                   'user': {'name': account_info[0]['nick_name'] if len(account_info[0]['nick_name']) > 0 else f'user{account_info[0]["id"][:8]}',
                                            'id': account_info[0]['id'],
                                            'balance': account_info[0]['balance'],
                                            'vip_level': account_info[0]['vip_level'],
                                            'locale': account_info[0]['locale'],
                                            },
                                   'expires_in': 3600
                                   })
        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False, 'message': "backend.api.error.default"})

