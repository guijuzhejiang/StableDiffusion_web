import asyncio

import async_timeout
import ujson
from sanic.request import Request
from websockets.exceptions import ConnectionClosed
from loguru import logger
from lib.celery_workshop.wokrshop import WorkShop
from operators import OperatorSD

sd_workshop = WorkShop(OperatorSD)


async def sd_genreate(request: Request, ws):
    try:
        logger.info(f"received generate request")
        user_id = request.args['user_id'][0]
        # check account
        account = (await \
            request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]

        while True:
            try:
                # recv params
                package = ujson.loads(await ws.recv())
                params = package['params']

                # cal prices
                cost_points = 1
                if package['mode'] == 'hires':
                    _output_width = int(params['output_width'])
                    _output_height = int(params['output_height'])

                    cost_points = 1
                    pixel_sum = _output_width + _output_height
                    if pixel_sum >= 3840:
                        cost_points += 1
                    if pixel_sum >= 5120:
                        cost_points += 1

                else:
                    cost_points *= int(int(params['batch_size']))

                # check balance
                account = await \
                request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute().data[0]
                if cost_points <= account['balance']:
                    # recv image
                    params['input_image'] = await ws.recv()
                    # proceed task
                    task_result = sd_workshop(**params)
                    while not task_result.ready():
                        await asyncio.sleep(1)
                        print('wait')
                    task_result = task_result.result
                    if task_result['success']:
                        data = request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                                            'amount': cost_points,
                                                                                            'is_plus': False,
                                                                                            'status': 1,
                                                                                            }).execute()
                        res = request.app.ctx.supabase_client.atable("account").update(
                            {"balance": account['balance'] - cost_points}).eq("id", user_id).execute().data

                else:
                    task_result = {'success': False, 'result': "余额不足"}

                await ws.send(ujson.dumps(task_result))
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass
    except Exception:
        print('fatal error')
        raise ConnectionClosed(1010, "fatal error")
