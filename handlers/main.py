import asyncio
import os.path
import random
import string
from datetime import datetime, timedelta
from io import BytesIO

import pytz
import ujson
from sanic.response import json as sanic_json, file_stream
from sanic.views import HTTPMethodView
from wechatpayv3 import WeChatPayType

from lib.celery_workshop.wokrshop import WorkShop
from lib.redis_mq import RedisMQ
from operators import OperatorSD
from utils.global_vars import CONFIG

sd_workshop = WorkShop(OperatorSD)
temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


class SDGenertae(HTTPMethodView):
    async def post(self, request):
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
            mode = request.form['mode'][0]
            params = ujson.loads(request.form['params'][0])
            user_id = request.form['user_id'][0]

            cost_points = 1

            if mode == 'hires':
                _output_width = int(params['output_width'])
                _output_height = int(params['output_height'])
                sum = _output_width + _output_height
                if sum >= 3840:
                    cost_points += 1
                if sum >= 5120:
                    cost_points += 1
            else:
                cost_points *= int(int(params['batch_size']))

            account = (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
            if cost_points <= account['balance']:
                task_result = sd_workshop(**request_form)
                print('wait')
                while not task_result.ready():
                    await asyncio.sleep(1)
                print('done')
                task_result = task_result.result
                if task_result['success']:
                    data = await request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                                        'amount': cost_points,
                                                                                        'is_plus': False,
                                                                                        'status': 1,
                                                                                        }).execute()
                    res = (await request.app.ctx.supabase_client.atable("account").update(
                        {"balance": account['balance']-cost_points}).eq("id", user_id).execute()).data

            else:
                task_result = {'success': False, 'result': "余额不足"}

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
        description = 'guiju_ai_model'
        unit_price = 4
        # 分为单位
        charge_points = int(request.form['amount'][0])
        total_price = charge_points * unit_price * 100 if account['access_level'] != 0 else 1
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


class Query(HTTPMethodView):
    async def post(self, request):
        user_id = request.form['user_id'][0]

        no_confirm_rows = (await request.app.ctx.supabase_client.atable("transaction").select("*").eq("user_id",
                                                                                          user_id).eq("is_plus", True).execute()).data

        # 计算未确认支付额
        pre_charge_amount = 0
        need_del_transaction = []
        for row in no_confirm_rows:
            if row['status'] == 0:
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
            return sanic_json({'balance': account['balance']+pre_charge_amount})


class ImageProvider(HTTPMethodView):
    async def get(self, request):
        return await file_stream(os.path.join(CONFIG['storage_dirpath']['user_dir'], request.args.get("imgpath")))
