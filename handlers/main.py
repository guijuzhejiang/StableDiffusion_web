import os.path

import traceback
import urllib.parse
from mimetypes import guess_type

import aiofile
import ujson
from sanic.compat import open_async
from sanic.response import json as sanic_json, file_stream, text, ResponseStream
from sanic.views import HTTPMethodView
from lib.celery_workshop.wokrshop import WorkShop
from lib.common.common_util import encrypt, generate_random_digits, uuid_to_number_string

from sanic.response import empty
from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from operators import OperatorSD
from utils.global_vars import CONFIG
sd_workshop = WorkShop(OperatorSD)
temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


class RevokeTask(HTTPMethodView):
    """
        撤销排队中的任务
    """
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            task_id = request.form['task_id'][0]
            request.app.ctx.sd_workshop.celery_app.control.revoke(task_id)
            await request.app.ctx.redis_session.lrem('celery_task_queue', count=1, value=task_id)
            # await request.app.ctx.redis_session.rpush('celery_task_revoked', task_id)
            await request.app.ctx.redis_session.set(task_id, 'revoke')
            return sanic_json({'success': True})
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'message': 'backend.api.error.default'})


class ImageProvider(HTTPMethodView):
    """
        fetch用户图片
    """
    async def get(self, request):
        if request.args.get("uid"):
            user_id = urllib.parse.unquote(request.args.get("uid"))

            category = request.args.get("category", 'model')
            dir_storage_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category)

            if category == 'account_avatar':
                fp = os.path.join(CONFIG['storage_dirpath']['user_account_avatar'], f"{user_id}.jpg")
                if not os.path.exists(fp):
                    return text('404 - Not Found', status=404)
            else:
                dir_user_path = os.path.join(dir_storage_path, user_id)
                fp = os.path.join(dir_user_path, request.args.get("imgpath"))
        else:
            dir_storage_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], 'hires')
            fp = os.path.join(dir_storage_path, request.args.get("imgpath"))

        return await file_stream(fp, chunk_size=1024)


class FetchUserHistory(HTTPMethodView):
    """
        查询用户生成历史
    """
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            category = request.args.get("category", 'model')
            origin = request.args.get("origin", 'https://www.imegaai.com')
            dir_storage_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category)

            dir_user_path = os.path.join(dir_storage_path, user_id)
            os.makedirs(dir_user_path, exist_ok=True)

            if 'imegaai' in origin:
                fixed_dn = origin.replace('www.', 'api.')
                if 'api.' not in fixed_dn:
                    protocol_txt = origin.split('://')[0]
                    fixed_dn = fixed_dn.replace(f'{protocol_txt}://', f'{protocol_txt}://api.')
            else:
                fixed_dn = origin

            # result = [ for img_fn in sorted(os.listdir(dir_user_path), reverse=True)]
            result = []
            user_gallery = (await request.app.ctx.supabase_client.atable("gallery").select("*").eq("user_id", user_id).order('instance_id', desc=True).execute()).data
            for img_fn in sorted(os.listdir(dir_user_path), reverse=True):
                url_fp = f"{'http://localhost:' + str(CONFIG['server']['port']) if CONFIG['local'] else f'{fixed_dn}/service'}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category={category}"
                user_item = {}
                for r in user_gallery:
                    if r['instance_id'] == img_fn:
                        user_item = r

                user_item['src'] = url_fp
                if 'category' not in user_item.keys():
                    user_item['category'] = category
                result.append(user_item)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.history'})
        else:
            return sanic_json({'success': True, 'result': result})


class FetchGallery(HTTPMethodView):
    """
        瀑布流图片
    """

    def unique_by_key(self, dicts, unique_key):
        seen = set()
        return [d for d in dicts if d[unique_key] not in seen and not seen.add(d[unique_key])]

    async def post(self, request):
        try:
            if 'query' in request.form:
                query_str = request.form['query'][0]

                result = []
                for query_item in query_str.split(','):
                    data = (await request.app.ctx.supabase_client.atable("gallery").select("*").is_("user_id", "NULL").like("prompt", f"%{query_item.strip()}%").order('instance_id', desc=True).execute()).data
                    result.extend(data)
                else:
                    result = self.unique_by_key(result, 'instance_id')

            elif 'id' in request.form:
                data_id = request.form['id'][0]
                result = (await request.app.ctx.supabase_client.atable("gallery").select("*").eq('id',data_id).order('instance_id', desc=True).execute()).data

            else:
                result = (await request.app.ctx.supabase_client.atable("gallery").select("*").is_("user_id", "NULL").order('instance_id', desc=True).execute()).data

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.history'})
        else:
            return sanic_json({'success': True, 'result': result})


class UserUpload(HTTPMethodView):
    """
        用户上传图片
    """
    async def post(self, request):
        try:
            user_id = urllib.parse.unquote(request.args.get("uid"))
            category = request.args.get("category")
            upload_image = request.files['upload_image'][0]
            # image_type = upload_image.type.split('/')[-1]
            if category:
                if category == 'account_avatar':
                    dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_account_avatar'])
                else:
                    dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category)
            else:
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_upload'])
            os.makedirs(dir_path, exist_ok=True)

            async with aiofile.async_open(os.path.join(dir_path, f"{user_id}.{'jpg' if category == 'account_avatar' else 'png'}"), 'wb') as file:
                await file.write(upload_image.body)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.upload'})
        else:
            return sanic_json({'success': True})

    async def options(self, request):
        return empty(status=204)


class UserEditNickname(HTTPMethodView):
    """
        用户上传图片
    """
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            new_nickname = request.form['nickname'][0]

            # nick_taken_res = (await request.app.ctx.supabase_client.atable("account").select('id').eq("nick_name", new_nickname).execute()).data
            # if len(nick_taken_res) > 0:
            #     return sanic_json({'success': False, 'message': 'backend.api.error.nickname'})
            # else:
            res = (await request.app.ctx.supabase_client.atable("account").update({'nick_name': new_nickname}).eq("id", user_id).execute()).data

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'message': 'backend.api.error.default'})
        else:
            return sanic_json({'success': True})


class FetchVideo(HTTPMethodView):
    """
        用户上传图片
    """

    def parse_range_header(self, range_header, file_size):
        # 解析Range参数
        start, end = range_header.split('=')[1].split('-')
        start = int(start)
        end = int(end) if end else file_size - 1

        # 确保范围在文件大小范围内
        start = min(max(start, 0), file_size - 1)
        end = min(end, file_size - 1)

        return start, end

    async def get(self, request):
        try:
            user_id = urllib.parse.unquote(request.args.get("uid"))
            category = request.args.get("category", 'sora')
            video_fn = request.args.get("path")
            dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category, user_id)

            file_path = os.path.join(dir_path, video_fn)
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            # 获取请求头Range参数
            range_header = request.headers.get("Range")
            headers = {}

            if range_header:
                # 解析Range参数
                start, end = self.parse_range_header(range_header, file_size)
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                headers['Content-Length'] = str(end - start + 1)
                chunk_size: int = 4096
                mime_type = None
                mime_type = mime_type or guess_type(file_path)[0] or "text/plain"

                # 这段代码就是拿 file_stream 里的源码改的，直接使用file_stream总是报错，有大佬帮忙解释并实现下吗
                async def _streaming_fn(response):
                    async with await open_async(file_path, mode="rb") as f:
                        await f.seek(start)
                        to_send = end - start + 1
                        while to_send > 0:
                            content = await f.read(min((to_send, chunk_size)))
                            if len(content) < 1:
                                break
                            to_send -= len(content)
                            await response.write(content)

                return ResponseStream(
                    streaming_fn=_streaming_fn,
                    status=status,
                    headers=headers,
                    content_type=mime_type,
                )
            else:
                return await file_stream(file_path, headers=headers)
            # return await file_stream(
            #     "/home/ray/Videos/simplescreenrecorder-2023-12-20_15.47.07.mp4",
            #     chunk_size=1024,
            #     mime_type="application/metalink4+xml",
            #     headers={
            #         "Content-Disposition": 'Attachment; filename="nicer_name.meta4"',
            #         "Content-Type": "application/metalink4+xml",
            #     },
            # )
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'message': 'backend.api.error.default'})


class SendCaptcha(HTTPMethodView):
    """
        发送手机验证码
    """
    async def post(self, request):
        try:
            exp_secs = 300
            phone = request.form['phone'][0]
            captcha = generate_random_digits()
            config = open_api_models.Config(
                # 必填，您的 AccessKey ID,
                access_key_id=os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'],
                # 必填，您的 AccessKey Secret,
                access_key_secret=os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
            )
            # Endpoint 请参考 https://api.aliyun.com/product/Dysmsapi
            config.endpoint = f'dysmsapi.aliyuncs.com'
            client = Dysmsapi20170525Client(config)
            send_sms_request = dysmsapi_20170525_models.SendSmsRequest(
                phone_numbers=phone,
                sign_name='幻景AI',
                template_code=CONFIG['aliyun']['sms']['template_code'] if phone[:3] == '+86' else CONFIG['aliyun']['sms']['global_template_code'],
                template_param=ujson.dumps({'code': captcha}),
            )
            try:
                # 复制代码运行请自行打印 API 的返回值
                res = await client.send_sms_with_options_async(send_sms_request, util_models.RuntimeOptions())
            except Exception as error:
                # 如有需要，请打印 error
                UtilClient.assert_as_string(error.message)
                print(error.message)
                return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})
            else:
                if res.body.code == 'OK':
                    await request.app.ctx.redis_session_sms.setex(phone, exp_secs, captcha)
                else:
                    return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})
        else:
            return sanic_json({'success': True})


class VerifyCaptcha(HTTPMethodView):
    """
        验证手机验证码
    """
    async def post(self, request):
        try:
            phone = request.form['phone'][0]
            captcha = request.form['captcha'][0]
            country = request.form['country'][0]

            redis_captcha = await request.app.ctx.redis_session_sms.get(phone)
            if redis_captcha:
                if redis_captcha == captcha:
                    # h = request.app.ctx.supabase_client.auth.headers
                    # response = await request.app.ctx.supabase_client.auth.async_api.http_client.get(
                    #     f"{request.app.ctx.supabase_client.auth.url}/admin/users?per_page=9999", headers=h)
                    # check_response(response)
                    # users = response.json().get("users")
                    id_res = (await request.app.ctx.supabase_client.atable("account").select("id").eq("phone", str(phone)).execute()).data
                    alike_email = f"{phone}@sms.com"
                    password = encrypt(phone+'guijutech').lower()

                    # 如果没有查询到则注册
                    if len(id_res) == 0:
                        try:
                            supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email=alike_email,
                                                                                                    password=password)
                            res = (await request.app.ctx.supabase_client.atable("account").update(
                                {"locale": country, 'phone': phone, 'nick_name': f'user{uuid_to_number_string(str(supabase_res.user.id))}'}).eq("id", str(supabase_res.user.id)).execute()).data

                        except Exception:
                            print(str(traceback.format_exc()))
                            return sanic_json({'success': False, 'message': "backend.api.error.register"})

                    account_info = (await request.app.ctx.supabase_client.atable("account").select(
                        "id,balance,locale,nick_name").eq("phone", phone).execute()).data
                    # 成功返回
                    return sanic_json({'success': True,
                                       'user': {'name': account_info[0]['nick_name'] if len(account_info[0]['nick_name']) > 0 else f'user{account_info[0]["id"][:8]}',
                                                'id': account_info[0]['id'],
                                                'balance': account_info[0]['balance'],
                                                'locale': account_info[0]['locale'],
                                                },
                                       'expires_in': 3600
                                       })

                else:
                    return sanic_json({'success': False, 'result': 'backend.api.error.wrong-captcha'})
            else:
                return sanic_json({'success': False, 'result': 'backend.api.error.no-captcha'})

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})
        else:
            return sanic_json({'success': True})

