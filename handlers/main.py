import asyncio

import ujson
from sanic.response import json as sanic_json
from sanic.views import HTTPMethodView

from lib.celery_workshop.wokrshop import WorkShop
from lib.redis_mq import RedisMQ
from operators import OperatorSD
from utils.global_vars import CONFIG

sd_workshop = WorkShop(OperatorSD)
temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


class SDGenertae(HTTPMethodView):
    async def post(self, request):
        if CONFIG['debug_mode']:
            redis_mq = RedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'],
                                             CONFIG['redis']['redis_mq'])

            task_result = await redis_mq.rpc_call(redis_mq.task_queue_name, request.form['params'][0])
            print(task_result)
            await redis_mq.close()
        else:
            task_result = sd_workshop(request.form['params'][0])
            while not task_result.ready():
                await asyncio.sleep(1)
                print('wait')
            task_result = task_result.result

        return sanic_json(task_result)


class SDHires(HTTPMethodView):
    async def post(self, request):
        task_result = sd_workshop()
        while not task_result.ready():
            await asyncio.sleep(1)
        return sanic_json({'result': task_result.result})
