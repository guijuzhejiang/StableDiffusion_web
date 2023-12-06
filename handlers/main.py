import asyncio
import os.path
import random
import string
import traceback
import urllib.parse
from datetime import datetime, timedelta, date
import aiofile
import httpx
import pytz
import ujson
from gotrue import check_response
from sanic.response import json as sanic_json, file_stream
from sanic.views import HTTPMethodView
from wechatpayv3 import WeChatPayType
from lib.celery_workshop.wokrshop import WorkShop
from lib.common.common_util import encrypt, generate_random_digits
from lib.redis_mq import RedisMQ
from lib.sanic_util.sanic_jinja2 import SanicJinja2
from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from operators import OperatorSD
from utils.global_vars import CONFIG
sd_workshop = WorkShop(OperatorSD)
temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


class SDGenertae(HTTPMethodView):
    async def post(self, request):
        try:
            # 解析表单参数
            request_form = dict(request.form)
            request_form['input_image'] = request.files['input_image']

            if CONFIG['debug_mode']:
                redis_mq = RedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'],
                                   CONFIG['redis']['redis_mq'])

                task_result = await redis_mq.rpc_call(redis_mq.task_queue_name, request_form)
                print(task_result)
                await redis_mq.close()

            else:
                mode = request_form['mode'][0]
                params = ujson.loads(request_form['params'][0])
                user_id = request_form['user_id'][0]

                cost_points = 10

                if mode == 'hires':
                    _output_width = int(params['output_width'])
                    _output_height = int(params['output_height'])
                    sum = _output_width + _output_height
                    if sum >= 2561:
                        cost_points = 16
                    if sum >= 4681:
                        cost_points = 20
                else:
                    cost_points *= int(params['batch_size'])

                account = \
                (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[
                    0]
                if cost_points <= account['balance']:
                    task_result = sd_workshop(**request_form)
                    print('wait')
                    while not task_result.ready():
                        await asyncio.sleep(1)
                    print('done')
                    task_result = task_result.result
                    if task_result['success']:
                        print('genreate success')
                        data = await request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                                                   'amount': cost_points,
                                                                                                   'is_plus': False,
                                                                                                   'status': 1,
                                                                                                   }).execute()
                        res = (await request.app.ctx.supabase_client.atable("account").update(
                            {"balance": account['balance'] - cost_points}).eq("id", user_id).execute()).data

                else:
                    task_result = {'success': False, 'result': "backend.balance.error.insufficient-balance"}
        except Exception:
            print(traceback.format_exc())
            task_result = {'success': False, 'result': "backend.api.error.default"}

        return sanic_json(task_result)


class SDHires(HTTPMethodView):
    async def post(self, request):
        task_result = sd_workshop()
        while not task_result.ready():
            await asyncio.sleep(1)
        return sanic_json({'result': task_result.result})


class Pay(HTTPMethodView):
    async def post(self, request):
        user_id = request.form['user_id'][0]
        account = (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
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


class RevokeTask(HTTPMethodView):
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


class QueryPayment(HTTPMethodView):
    async def post(self, request):
        out_trade_no = request.form['out_trade_no'][0]
        code, message = request.app.ctx.wxpay.query(
            out_trade_no=out_trade_no,
        )

        res = ujson.loads(message)

        return sanic_json({'success': code == 200, 'message': res})


class WeChatLogin(HTTPMethodView):
    async def get(self, request):
        state = request.args.get('state')
        code = request.args.get('code')
        # users = await request.app.ctx.supabase_client.auth.async_list_users()
        # print(users)
        # supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email="ezzmai@teal.com", password="pa6666ssword")
        # print(supabase_res)
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
                print(
                    f'https://api.weixin.qq.com/sns/oauth2/access_token?appid={CONFIG["wechatlogin"]["appid"]}&secret={CONFIG["wechatlogin"]["secret"]}&code={code}&grant_type=authorization_code')
                # 检查响应状态码
                if response.status_code == 200:
                    # 获取响应的 JSON 数据
                    wechat_data = response.json()
                    print(wechat_data)
                    email = f"{wechat_data['openid']}@wechat.com".lower()
                    password = encrypt(str({wechat_data['openid']}).lower())
                    result_user = {'username': email, 'password': password}

                    # 获取微信头像
                    response_uinfo = await client.get(
                        f'https://api.weixin.qq.com/sns/userinfo?access_token={wechat_data["access_token"]}&openid={wechat_data["openid"]}')
                    if response_uinfo.status_code == 200:
                        wechat_uinfo = response_uinfo.json()
                        result_user['avatar'] = wechat_uinfo['headimgurl']
                        result_user['name'] = wechat_uinfo['nickname']

                    # supabase 检查有没有改用户，没有就注册
                    # users = await request.app.ctx.supabase_client.auth.async_list_users()
                    h = request.app.ctx.supabase_client.auth.headers
                    response = await request.app.ctx.supabase_client.auth.async_api.http_client.get(
                        f"{request.app.ctx.supabase_client.auth.url}/admin/users?per_page=9999", headers=h)
                    check_response(response)
                    users = response.json().get("users")
                    # if users is None:
                    #     return sanic_json({'success': False, 'message': "登录失败"})
                    if not isinstance(users, list):
                        return sanic_json({'success': False, 'message': "backend.api.error.default"})

                    users_email = [u['email'] for u in users]
                    if email not in users_email:
                        try:
                            supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email=email,
                                                                                               password=password)
                        except Exception:
                            print(str(traceback.format_exc()))
                            return sanic_json({'success': False, 'message': "backend.api.error.register"})

                    # 成功返回
                    return sanic_json({'success': True, 'user': result_user})
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    return sanic_json({'success': False, 'message': "登录失败"})
        except Exception:
            print(str(traceback.format_exc()))
            return sanic_json({'success': False, 'message': "backend.api.error.default"})


class Query(HTTPMethodView):
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
    async def get(self, request):
        if request.args.get("uid"):
            user_id = urllib.parse.unquote(request.args.get("uid"))

            category = request.args.get("category", 'model')
            # if category == 'hair':
            #     dir_storage_path = CONFIG['storage_dirpath']['user_hair_dir']
            # elif category == 'mirror':
            #     dir_storage_path = CONFIG['storage_dirpath']['user_mirror_dir']
            # elif category == 'avatar':
            #     dir_storage_path = CONFIG['storage_dirpath']['user_avatar_dir']
            # else:
            dir_storage_path = CONFIG['storage_dirpath'][f'user_{category}_dir']
            dir_user_path = os.path.join(dir_storage_path, user_id)

            fp = os.path.join(dir_user_path, request.args.get("imgpath"))
        else:
            fp = os.path.join(CONFIG['storage_dirpath']['hires_dir'], request.args.get("imgpath"))

        return await file_stream(fp)


class FetchUserHistory(HTTPMethodView):
    async def post(self, request):
        try:
            user_id = request.form['user_id'][0]
            category = request.args.get("category", 'model')
            # if category == 'hair':
            #     dir_storage_path = CONFIG['storage_dirpath']['user_hair_dir']
            # elif category == 'avatar':
            #     dir_storage_path = CONFIG['storage_dirpath']['user_avatar_dir']
            # elif category == 'mirror':
            #     dir_storage_path = CONFIG['storage_dirpath']['user_mirror_dir']
            # else:
            #     dir_storage_path = CONFIG['storage_dirpath']['user_dir']
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
    async def post(self, request):
        try:
            user_id = urllib.parse.unquote(request.args.get("uid"))
            category = request.args.get("category")
            upload_image = request.files['upload_image'][0]
            image_type = upload_image.type.split('/')[-1]
            if category:
                dir_path = os.path.join(CONFIG['storage_dirpath'][f'user_{category}_upload'])
            else:
                dir_path = os.path.join(CONFIG['storage_dirpath']['user_upload'])
            os.makedirs(dir_path, exist_ok=True)

            async with aiofile.async_open(os.path.join(dir_path, f"{user_id}.png"), 'wb') as file:
                await file.write(upload_image.body)

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.upload'})
        else:
            return sanic_json({'success': True})


class SendCaptcha(HTTPMethodView):
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
    async def post(self, request):
        try:
            phone = request.form['phone'][0]
            captcha = request.form['captcha'][0]
            country = request.form['country'][0]

            redis_captcha = await request.app.ctx.redis_session_sms.get(phone)
            if redis_captcha:
                if redis_captcha == captcha:
                    h = request.app.ctx.supabase_client.auth.headers
                    response = await request.app.ctx.supabase_client.auth.async_api.http_client.get(
                        f"{request.app.ctx.supabase_client.auth.url}/admin/users?per_page=9999", headers=h)
                    check_response(response)
                    users = response.json().get("users")

                    alike_email = f"{phone}@sms.com"
                    password = encrypt(phone+'guijutech').lower()

                    if not isinstance(users, list):
                        return sanic_json({'success': False, 'message': "backend.api.error.default"})

                    users_email = [u['email'] for u in users]
                    if alike_email not in users_email:
                        try:
                            supabase_res = await request.app.ctx.supabase_client.auth.async_sign_up(email=alike_email,
                                                                                                    password=password)
                            res = (await request.app.ctx.supabase_client.atable("account").update(
                                {"locale": country}).eq("id", str(supabase_res.user.id)).execute()).data

                            return sanic_json({'success': True, 'username': alike_email, 'password': password})
                        except Exception:
                            print(str(traceback.format_exc()))
                            return sanic_json({'success': False, 'message': "backend.api.error.register"})
                    else:
                        return sanic_json({'success': True, 'username': alike_email, 'password': password})

                else:
                    return sanic_json({'success': False, 'result': 'backend.api.error.wrong-captcha'})
            else:
                return sanic_json({'success': False, 'result': 'backend.api.error.no-captcha'})

        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.send-captcha'})
        else:
            return sanic_json({'success': True})