# coding=utf-8
# @Time : 2023/12/29 下午4:18
# @File : login.py
import asyncio
import traceback

import httpx
import aiofile
import os

import ujson
from sanic.response import json as sanic_json, file_stream, empty
from sanic.views import HTTPMethodView

from lib.common.common_util import encrypt
from lib.sanic_util.sanic_jinja2 import SanicJinja2
from google.oauth2 import id_token
from google.auth.transport import requests
from requests import Session

from utils.global_vars import CONFIG


class SDGen(HTTPMethodView):
    """
        SD
    """
    async def post(self, request):
        try:
            prompt = request.form['prompt'][0]
            if prompt[0] == '"':
                prompt = prompt[1:]
            if prompt[-1] == '"':
                prompt = prompt[:-1]
            user_id = request.form['user_id'][0]
            msgs_length = request.form['mlen'][0]
            chat_id = request.form['chat_id'][0]
            dp = f'zs/bg_buffer'
            os.makedirs(dp, exist_ok=True)
            format_package = {'mode': ['text2image'],
                              'user_id': user_id,
                              'params': [ujson.dumps(
                                  {
                                      "batch_size": 1,
                                      "style": 0,
                                      "width": 1024,
                                      "height": 1024,
                                      "prompt": prompt,
                                      "sim": 0.7,
                                      "mode": 'text2image',
                                      "translate": False,
                                      "mlen": msgs_length,
                                      "chat_id": chat_id,
                                  }
                              )],
                              'origin': 'zs.guijutech.com',
                              'input_image': ''}
            task_result = request.app.ctx.sd_workshop(**format_package)
            task_id = str(task_result)
            pending_task_id = await request.app.ctx.redis_session.get(user_id)
            if pending_task_id:
                try:
                    request.app.ctx.sd_workshop.celery_app.control.revoke(pending_task_id)
                except Exception:
                    pass
            await request.app.ctx.redis_session.set(user_id, task_id)
            while not task_result.ready():
                if task_result.state == 'SUCCESS':
                    break

                elif task_result.state == 'REVOKED':
                    break

                await asyncio.sleep(0.5)
            # 成功返回
            return sanic_json(task_result.result)

        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False})


class SDBgProvider(HTTPMethodView):
    """
        SD
    """
    async def get(self, request):
        try:
            user_id = request.args.get("uid")
            chat_id = request.args.get("cid")
            msgs_len = request.args.get("mlen")
            check = request.args.get("check", False)

            dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], 'learninglanggptbg', user_id, chat_id)

            bg_fp = os.path.join(dir_path, f"{msgs_len}.png")
            if os.path.exists(bg_fp):
                if check:
                    return empty(status=200)
                else:
                    # 成功返回
                    return await file_stream(bg_fp, chunk_size=1024)
            else:
                return empty(status=404)

        except Exception:
            print(str(traceback.format_exc()))
            return empty(status=404)
