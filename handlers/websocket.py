import asyncio
import datetime
import os
import traceback

from urllib.parse import urlparse, parse_qs

import ujson
from sanic.request import Request
from loguru import logger
from lib.common.common_util import logging
from utils.global_vars import CONFIG


async def sd_genreate(request: Request, ws):
    try:
        logger.info(f"received generate request")
        user_id = request.args['user_id'][0]
        # check account
        account = \
        (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data
        if len(account) <= 0:
            raise Exception

        try:
            # recv params
            raw_msg = await ws.recv()

            package = ujson.loads(raw_msg)
            params = package['params']
            format_package = {'mode': [package['mode']], 'user_id': [package['user_id']], 'params': [ujson.dumps(params)], 'input_image': ''}

            # cal prices
            cost_points = 10
            if package['mode'] == 'hires':
                if params.get('hires_times'):
                    cost_points = 5 if int(params['hires_times']) == 2 else 10
                else:
                    _output_width = int(params['output_width'])
                    _output_height = int(params['output_height'])

                    cost_points = 5
                    pixel_sum = _output_width + _output_height
                    if pixel_sum >= 2561:
                        cost_points = 10
                    if pixel_sum >= 4681:
                        cost_points = 15

            elif package['mode'] == 'avatar':
                batch_size = int(params['batch_size'])

                if batch_size == 1:
                    cost_points = 5
                elif batch_size == 2:
                    cost_points = 8

            elif package['mode'] == 'mirror':
                batch_size = int(params['batch_size'])

                if batch_size == 1:
                    cost_points = 5
                elif batch_size == 2:
                    cost_points = 8

            elif package['mode'] == 'hair':
                batch_size = int(params['batch_size'])

                if batch_size == 1:
                    cost_points = 5
                elif batch_size == 2:
                    cost_points = 8
            else:
                batch_size = int(params['batch_size'])

                if batch_size == 1:
                    cost_points = 10
                elif batch_size == 2:
                    cost_points = 15
                else:
                    cost_points = 18

            # check balance
            account = (await request.app.ctx.supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
            buf_result = {'success': True, 'result': None, 'act': None, 'type': package['mode']}
            if cost_points <= account['balance']:
                # recv image
                if package['mode'] == 'hires':
                    parsed_url = urlparse(package['chosen_image'])
                    # 获取查询参数
                    query_params = parse_qs(parsed_url.query)

                    category = query_params['category'][0]
                    if category == 'hair':
                        dir_storage_path = CONFIG['storage_dirpath']['user_hair_dir']
                    elif category == 'mirror':
                        dir_storage_path = CONFIG['storage_dirpath']['user_mirror_dir']
                    else:
                        dir_storage_path = CONFIG['storage_dirpath']['user_dir']
                    img_fp = os.path.join(dir_storage_path, query_params['uid'][0], query_params['imgpath'][0])
                    format_package['input_image'] = img_fp
                else:
                    format_package['input_image'] = os.path.join(CONFIG['storage_dirpath']['user_upload'], f"{user_id}.png")

                # send task
                task_result = request.app.ctx.sd_workshop(**format_package)
                await ws.send(ujson.dumps({'success': True, 'result': str(task_result), 'act': 'save_task_id', 'type': package['mode']}))
                await request.app.ctx.redis_session.rpush('celery_task_queue', str(task_result))

                print('wait')
                while not task_result.ready():
                    # print(task_result.state)
                    if task_result.state == 'PROGRESS':
                        buf_result['act'] = 'show_progress'
                        if task_result.info is not None:
                            try:
                                buf_result['result'] = task_result.info['progress']
                                await ws.send(ujson.dumps(buf_result))
                            except Exception:
                                buf_result['result'] = 99
                                await ws.send(ujson.dumps(buf_result))

                    elif task_result.state == 'PENDING':
                        buf_result['act'] = 'show_queue'
                        try:
                            queue_list = await request.app.ctx.redis_session.lrange('celery_task_queue', 0, -1)
                            if queue_list:
                                if str(task_result) in queue_list:
                                    buf_result['result'] = f"第{queue_list.index(str(task_result)) + 1}位"
                                else:
                                    buf_result['result'] = f"..."
                            else:
                                buf_result['result'] = f"..."
                            await ws.send(ujson.dumps(buf_result))

                        except Exception:
                            print("fuck1")
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

                    elif task_result.state == 'REVOKED':
                        break

                    await asyncio.sleep(0.5)
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
                task_result = {'success': False, 'result': "backend.api.error.insufficient-balance"}

            await ws.send(ujson.dumps(task_result))

        except asyncio.TimeoutError:
            print("fuck2")

    except Exception:
        print("fuck3")

        logging(
            f"[websocket fatal error][{datetime.datetime.now()}]:"
            f"{traceback.format_exc()}",
            f"logs/error.log", print_msg=True)
        await ws.send(ujson.dumps({'success': False, 'result': "backend.api.error.default"}))
