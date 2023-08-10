import base64
import datetime
import json
import os
import traceback

import aiofiles
import cv2
from sanic.response import json as sanic_json
from sanic.response import text, redirect
from sanic.views import HTTPMethodView

from lib.common.common_util import logging, generate_random
from lib.sanic_util.sanic_jinja2 import SanicJinja2
from lib.sanic_util.sanic_jwt_auth import JWTAuth
from utils.global_vars import CONFIG, UTF_TD
from utils.template_resources import multi_langs


class Upload(HTTPMethodView):
    """
    ocr 识别入口
    """
    async def post(self, request):
        logging(
            f"[Upload]|{request.protocol}|{request.method}]"
            f"recv POST",
            f"logs/info.log", print_msg=CONFIG['debug_mode'])
        # ----------params
        try:
            upload_start_time = request.form.get("start_record_time", None)
            if upload_start_time is None:
                upload_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')

            # 文件格式
            # file_type = request.form.get('file_type', None)
            # 文件标识id
            sid = request.form.get('sid', None)
            if sid is None:
                sid = generate_random(10)
            # lang = request.form.get('lang', 'jp')
            # 图片类型， 表格或身份证
            img_type = request.form.get('img_type', None)
            # 是否画效果图
            bool_draw_img = request.form.get('bool_draw_img', 'false') == 'true'
            # 是否排序
            bool_sort = request.form.get('bool_sort', 'true') == 'true'
            # 是否返回原图
            bool_return_img = request.form.get('bool_return_img', 'false') == 'true'
            # 图片数据
            img_data = request.files.get('img_data').body
            file_type = request.files.get('img_data').name.split('.')[-1]
            if file_type.lower() not in ['jpg', 'jpeg', 'png', 'pdf', 'bmp']:
                return sanic_json({'error': f'.{file_type} file are not supported'})
            #
            mode = request.form.get('mode', 'ocr')
            call_queue_name = request.app.ctx.redis_mq.task_queue_name

            if type(img_data) != bytes:
                raise Exception
        except Exception as err:
            logging(
                f"[{__file__}-Upload|get params][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

            return sanic_json({'error': 'get params error'})

        # save upload file
        try:
            fn = f"{sid}.{upload_start_time}.{file_type}"
            client_upload_fp = os.path.join(CONFIG['storage_dirpath'].get('user_dir'), 'upload', fn)
            async with aiofiles.open(client_upload_fp, 'wb') as f:
                await f.write(img_data)

            # log info
            time_point_end = datetime.datetime.now()
        except Exception:
            logging(
                f"[{__file__}-Upload|save file][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

            return sanic_json({'error': 'save upload file error'})

        # redis_mq publishing
        res_dict = {'error': 'redis mq publish failed'}
        try:
            # request.app.ctx.redis_mq.connect()
            # await asyncio.sleep(0.8*random.random())
            res_dict = await request.app.ctx.redis_mq.rpc_call(call_queue_name,
                                                               data_fp=client_upload_fp,
                                                               bool_draw_img=bool_draw_img,
                                                               bool_sort=bool_sort,
                                                               img_type=img_type,
                                                               bool_return_img=bool_return_img,
                                                               operation=mode)

        except Exception:
            logging(
                f"[upload|post redis mq][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        finally:
            pass

        if 'ori_imgs' in res_dict.keys():
            img_paths = res_dict['ori_imgs']
            res_dict['ori_imgs'] = [
                base64.b64encode(cv2.imencode(f".jpg", cv2.imread(x)[:, :, ::-1])[1].tobytes()).decode('utf8')
                for x in img_paths]
        return sanic_json(res_dict['json_result'])


class Index(HTTPMethodView):
    # @JWTAuth.jwt_protected("test")
    @JWTAuth.jwt_login_required
    async def get(self, request):
        return redirect("main")


class Main(HTTPMethodView):
    # @JWTAuth.jwt_protected("test")
    @JWTAuth.jwt_login_required
    async def get(self, request):
        params = {i: multi_langs[i][CONFIG['ocr']['lang']] for i in
                  ["ocr_upload_btn", "ocr_upload_btn_note", "tts_btn"]}
        params['version'] = str(datetime.datetime.timestamp(datetime.datetime.now()))
        return await SanicJinja2.template_render_async("index.html", **params)


class Table(HTTPMethodView):
    @JWTAuth.jwt_login_required
    async def get(self, request):
        params = {i: multi_langs[i][CONFIG['ocr']['lang']] for i in
                                                          ["ocr_upload_btn", "ocr_upload_btn_note", "tts_btn"]}
        if CONFIG['ocr']['lang'] == 'ch':
            params['tts_hide'] = True
        params['version'] = str(datetime.datetime.timestamp(datetime.datetime.now()))
        return await SanicJinja2.template_render_async("table.html", **params)


class Tts(HTTPMethodView):
    async def post(self, request):
        logging(
            f"[Tts]|{request.protocol}|{request.method}]"
            f"recv POST",
            f"logs/info.log", print_msg=CONFIG['debug_mode'])
        # ----------params
        try:
            text = request.form.get('text', None)
            sid = request.form.get('sid', None)
            call_queue_name = request.app.ctx.redis_mq.task_queue_name
            operation = 'tts'
            res_dict = await request.app.ctx.redis_mq.rpc_call(call_queue_name,
                                                               text=text,
                                                               sid=sid,
                                                               operation=operation)
        except Exception as err:
            logging(
                f"[{__file__}-Upload|get params][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

            return sanic_json({'error': 'get params error'})

        else:
            return sanic_json(res_dict)
