import asyncio

import async_timeout
import ujson
from celery.result import AsyncResult
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
                if cost_points <= account['balance']:
                    print("ready for proceed generate")
                    await ws.send(ujson.dumps({'success': True, 'result': '', 'act': 'send_image'}))
                    # recv image
                    format_package['input_image'][0].append(await ws.recv())
                    # proceed task
                    task_result = sd_workshop(**format_package)
                    print('wait')
                    while not task_result.ready():
                        print(task_result.state)
                        if task_result.state == 'PROGRESS':
                            if task_result.info is not None:
                                try:
                                    await ws.send(ujson.dumps({'success': True, 'result': task_result.info['progress'], 'act': f"show_{package['mode']}_progress"}))
                                except Exception:
                                    await ws.send(ujson.dumps({'success': True, 'result': 99, 'act': f"show_{package['mode']}_progress"}))
                                    break
                        elif task_result.state == 'PENDING':
                            try:
                                queue_list = task_result.app.control.inspect().reserved()[f'{sd_workshop.op.__name__}_worker']

                                get_success = False
                                for index, q in enumerate(queue_list):
                                    if str(task_result) == q['id']:
                                        get_success = True
                                        await ws.send(ujson.dumps({'success': True, 'result': index+1,
                                                                   'act': f"show_{package['mode']}_queue"}))
                                else:
                                    if not get_success:
                                        await ws.send(ujson.dumps({'success': True, 'result': len(queue_list),
                                                                   'act': f"show_{package['mode']}_queue"}))

                            except Exception:
                                await ws.send(ujson.dumps({'success': True, 'result': '...',
                                                           'act': f"show_{package['mode']}_queue"}))

                        elif task_result.state == 'SUCCESS':
                            await ws.send(ujson.dumps({'success': True, 'result': 100, 'act': f"show_{package['mode']}_progress"}))
                            break
                        await asyncio.sleep(1)
                    print('done.')
                    task_result = task_result.result
                    task_result['act'] = f"show_{package['mode']}_result"
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
        print('fatal error')
        await ws.send(ujson.dumps({'success': False, 'result': "fatal error"}))
