import os.path

import traceback
import urllib.parse
from datetime import datetime, timedelta, date
import aiofile
import ujson
from sanic.response import json as sanic_json, file_stream
from sanic.views import HTTPMethodView
from lib.celery_workshop.wokrshop import WorkShop
from lib.common.common_util import encrypt, generate_random_digits, uuid_to_number_string
import pytz
from gotrue import check_response
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


class QueryDiscount(HTTPMethodView):
    """
        查询当前可用优惠
    """
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            transaction_data = (await request.app.ctx.supabase_client.atable("transaction").select("*").eq("user_id", user_id).eq("is_plus", True).eq("status", 1).execute())
            first_charge = len(transaction_data.data) == 0

            res = []
            if first_charge:
                res.append(['首次充值享8折优惠', 0.8])

            start_date = date(2023, 12, 1)
            end_date = date(2023, 12, 29)
            if start_date <= date.today() <= end_date:
                res.append(['十二月优惠礼6折', 0.6])

            if len(res) > 1:
                res.append(['多个折扣可叠加', 1])

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
        no_confirm_rows = (await request.app.ctx.supabase_client.atable("transaction").select("*").eq("user_id", user_id).eq("status", 0).eq("is_plus", True).execute()).data

        # 计算未确认支付额
        pre_charge_amount = 0
        need_del_transaction = []
        for row in no_confirm_rows:
            code, message = request.app.ctx.wxpay.query(
                # transaction_id='demo-transation-id'
                out_trade_no=row['out_trade_no']
            )
            trade_message = ujson.loads(message)

            if code == 200 and trade_message['trade_state'] == 'SUCCESS':
                data = (await request.app.ctx.supabase_client.atable("transaction").update(
                    {"status": 1}).eq("id", row['id']).eq("is_plus", True).execute()).data
                if len(data) == 0:
                    print(row['id'] + " update transaction false")
                pre_charge_amount += row['amount']
            else:
                iso_created_at = row['created_at'].split('.')[0] + '+' + row['created_at'].split('+')[-1]
                if (datetime.now(pytz.UTC) - datetime.fromisoformat(iso_created_at).replace(tzinfo=pytz.UTC)) >= timedelta(minutes=15):
                    need_del_transaction.append(row['id'])

        account = (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
        if pre_charge_amount > 0:
            data = (await request.app.ctx.supabase_client.atable("account").update(
                        {"balance": account['balance']+pre_charge_amount}).eq("id", account['id']).execute()).data
        if len(need_del_transaction):
            for del_target in need_del_transaction:
                data = (await request.app.ctx.supabase_client.atable("transaction").delete().eq("id", del_target).execute()).data

        if hasattr(request.form, 'out_trade_no'):
            out_trade_no = request.form['out_trade_no'][0]
            code, message = request.app.ctx.wxpay.query(
                out_trade_no=out_trade_no
            )

            return sanic_json({'success': code == 200, 'balance': account['balance']+pre_charge_amount})
        else:
            return sanic_json({'success': True, 'balance': account['balance']+pre_charge_amount})


class ImageProvider(HTTPMethodView):
    """
        fetch用户图片
    """
    async def get(self, request):
        if request.args.get("uid"):
            user_id = urllib.parse.unquote(request.args.get("uid"))

            category = request.args.get("category", 'model')
            dir_storage_path = CONFIG['storage_dirpath'][f'user_{category}_dir']

            if category == 'account_avatar':
                fp = os.path.join(dir_storage_path, f"{user_id}.jpg")
            else:
                dir_user_path = os.path.join(dir_storage_path, user_id)
                fp = os.path.join(dir_user_path, request.args.get("imgpath"))
        else:
            fp = os.path.join(CONFIG['storage_dirpath']['hires_dir'], request.args.get("imgpath"))

        return await file_stream(fp)


class FetchUserHistory(HTTPMethodView):
    """
        查询用户生成历史
    """
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            category = request.args.get("category", 'model')
            dir_storage_path = CONFIG['storage_dirpath'][f'user_{category}_dir']
            dir_user_path = os.path.join(dir_storage_path, user_id)
            os.makedirs(dir_user_path, exist_ok=True)

            result = [f"{'http://192.168.110.8:' + str(CONFIG['server']['port']) if CONFIG['local'] else CONFIG['server']['client_access_url']}/user/image/fetch?imgpath={img_fn}&uid={urllib.parse.quote(user_id)}&category={category}" for img_fn in sorted(os.listdir(dir_user_path), reverse=True)]
            if len(result) < 10:
                for i in range(10-len(result)):
                    result.append('')

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
            image_type = upload_image.type.split('/')[-1]
            if category:
                if category == 'account_avatar':
                    dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_account_avatar_dir'])
                else:
                    dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_{category}_upload'])
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


class UserEditNickname(HTTPMethodView):
    """
        用户上传图片
    """
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            new_nickname = request.form['nickname'][0]

            nick_taken_res = (await request.app.ctx.supabase_client.atable("account").select('id').eq("nick_name", new_nickname).execute()).data
            if len(nick_taken_res) > 0:
                return sanic_json({'success': False, 'message': 'backend.api.error.nickname'})
            else:
                res = (await request.app.ctx.supabase_client.atable("account").update({'nick_name': new_nickname}).eq("id", user_id).execute()).data

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'message': 'backend.api.error.default'})
        else:
            return sanic_json({'success': True})


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
                template_code=CONFIG['aliyun']['sms']['template_code'],
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
                                       'user': {'name': account_info[0]['nick_name'],
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


class PasswordLogin(HTTPMethodView):
    """
        密码登录
    """
    async def post(self, request):
        try:
            phone = request.form['phone'][0]
            password = request.form['password'][0]

            supabase_res = await request.app.ctx.supabase_client.auth.async_sign_in(email=phone, password=password)
            account_info = (await request.app.ctx.supabase_client.atable("account").select(
                "id,balance,locale,nick_name").eq("id", supabase_res.user.id).execute()).data
            return sanic_json({'success': True,
                               'user': {'name': account_info[0]['nick_name'],
                                        'id': account_info[0]['id'],
                                        'balance': account_info[0]['balance'],
                                        'locale': account_info[0]['locale'],
                                        },
                               'expires_in': 3600
                               })
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})
