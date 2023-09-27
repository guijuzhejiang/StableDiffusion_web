import asyncio
import datetime
import traceback

import async_timeout
import ujson
from celery.result import AsyncResult
from sanic.request import Request
from websockets.exceptions import ConnectionClosed
from loguru import logger
from lib.celery_workshop.wokrshop import WorkShop
from lib.common.common_util import logging
from operators import OperatorSD

sd_workshop = WorkShop(OperatorSD)


async def sd_genreate(request: Request, ws):
    try:
        logger.info(f"received generate request")
        user_id = request.args['user_id'][0]
        # check account
        account = \
        (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]

        while True:
            try:
                # recv params
                raw_msg = await ws.recv()
                package = ujson.loads(raw_msg)
                params = package['params']
                format_package = {'mode': [package['mode']], 'user_id': [package['user_id']], 'params': [ujson.dumps(params)], 'input_image': [['']]}

                # cal prices
                cost_points = 10
                if package['mode'] == 'hires':
                    _output_width = int(params['output_width'])
                    _output_height = int(params['output_height'])

                    cost_points = 1
                    pixel_sum = _output_width + _output_height
                    if pixel_sum >= 2561:
                        cost_points = 16
                    if pixel_sum >= 4681:
                        cost_points = 20
                else:
                    cost_points *= int(int(params['batch_size']))

                # check balance
                account = \
                (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[
                    0]
                buf_result = {'success': True, 'result': None, 'act': None, 'type': package['mode']}
                if cost_points <= account['balance']:
                    print("ready for proceed generate")
                    buf_result['act'] = 'send_image'
                    await ws.send(ujson.dumps(buf_result))
                    # recv image
                    format_package['input_image'][0].append(await ws.recv())
                    # proceed task
                    task_result = sd_workshop(**format_package)
                    print('wait')
                    while not task_result.ready():
                        print(task_result.state)
                        if task_result.state == 'PROGRESS':
                            buf_result['act'] = 'show_progress'
                            if task_result.info is not None:
                                try:
                                    buf_result['result'] = task_result.info['progress']
                                    await ws.send(ujson.dumps(buf_result))
                                except Exception:
                                    buf_result['result'] = 99
                                    await ws.send(ujson.dumps(buf_result))
                                    break
                        elif task_result.state == 'PENDING':
                            buf_result['act'] = 'show_queue'
                            try:
                                queue_list = task_result.app.control.inspect().reserved()[f'{sd_workshop.op.__name__}_worker']
                                get_success = False
                                for index, q in enumerate(queue_list):
                                    if str(task_result) == q['id']:
                                        get_success = True
                                        buf_result['result'] = f"{index+1}/{len(queue_list)}"
                                        await ws.send(ujson.dumps(buf_result))
                                else:
                                    if not get_success:
                                        buf_result['result'] = f"..."
                                        await ws.send(ujson.dumps(buf_result))

                            except Exception:
                                logging(
                                    f"[websocket fatal error][{datetime.datetime.now()}]:"
                                    f"{traceback.format_exc()}",
                                    f"logs/error.log", print_msg=True)
                                buf_result['result'] = f"..."
                                await ws.send(ujson.dumps(buf_result))

                        elif task_result.state == 'SUCCESS':
                            buf_result['act'] = 'show_progress'
                            buf_result['result'] = 100
                            await ws.send(ujson.dumps(buf_result))
                            break
                        await asyncio.sleep(1)
                    print('done.')
                    task_result = task_result.result
                    task_result['act'] = f"show_result"
                    task_result['type'] = package['mode']
                    if task_result['success']:
                        data = await request.app.ctx.supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                                                   'amount': cost_points,
                                                                                                   'is_plus': False,
                                                                                                   'status': 1,
                                                                                                   }).execute()
                        res = (await request.app.ctx.supabase_client.atable("account").update(
                            {"balance": account['balance'] - cost_points}).eq("id", user_id).execute()).data

                else:
                    task_result = {'success': False, 'result': "余额不足"}

                await ws.send(ujson.dumps(task_result))
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass
    except Exception:
        logging(
            f"[websocket fatal error][{datetime.datetime.now()}]:"
            f"{traceback.format_exc()}",
            f"logs/error.log", print_msg=True)
        await ws.send(ujson.dumps({'success': False, 'result': "fatal error"}))
