import datetime
import traceback

from sanic.views import HTTPMethodView

from lib.common.common_util import logging
from lib.sanic_util.sanic_jinja2 import SanicJinja2
from lib.sanic_util.sanic_jwt_auth import JWTAuth
from utils.global_vars import CONFIG
from sanic.response import json as sanic_json
from sanic.response import redirect, text

temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


class Login(HTTPMethodView):
    async def get(self, request):
        params = {}
        params['version'] = str(datetime.datetime.timestamp(datetime.datetime.now()))
        return await SanicJinja2.template_render_async("login.html", **params)

    async def post(self, request):
        # ----------params
        try:
            uid = request.form.get('user_id', None)
            upwd = request.form.get('user_password', None)
            login_datetime = request.form.get('login_datetime', None)
            logging(
                f"[Login]|{request.protocol}|{request.method}]"
                f"user login : {uid} - {upwd}",
                f"logs/info.log", print_msg=CONFIG['debug_mode'])

            params = {}
            params['version'] = str(datetime.datetime.timestamp(datetime.datetime.now()))
            if uid is not None and upwd is not None and uid == upwd and uid in temp_udb:
                response = redirect('main')
                response.cookies['PTOKEN'] = JWTAuth.web_login({'uid': uid})
                response.cookies['PTOKEN']['expires'] = datetime.datetime.strptime(login_datetime, '%Y-%m-%d_%H:%M:%S')
                # response.cookies['PTOKEN']['domain'] = request.host.replace('www', '')
                return response
            else:
                params['msg'] = 'wrong user id or password'
                return await SanicJinja2.template_render_async("login.html", **params)

        except Exception as err:
            logging(
                f"[{__file__}-Upload|get params][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

            return sanic_json({'error': 'fatal error'})


class Logout(HTTPMethodView):
    async def get(self, request):
        response = redirect('login')
        response.cookies['PTOKEN'] = ''
        response.cookies['PTOKEN']["max-age"] = 0
        del response.cookies['PTOKEN']
        return response
