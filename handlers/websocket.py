import asyncio
import datetime
import os
import traceback

from urllib.parse import urlparse, parse_qs

import ujson
from sanic.request import Request
from loguru import logger

from handlers.qinghua import auth_devices
from lib.common.common_util import logging
from utils.global_vars import CONFIG


async def sd_genreate(request: Request, ws):
    try:
        logger.info(f"received generate request")
        user_id = request.args['user_id'][0]
        # check account
        account = \
        (await request.app.ctx.supabase_client.table("account").select("*").eq("id", user_id).execute()).data
        if len(account) <= 0:
            raise Exception

        try:
            # recv params
            raw_msg = await ws.recv()

            package = ujson.loads(raw_msg)
            params = package['params']
            format_package = {'mode': [package['mode']],
                              'user_id': [package['user_id']],
                              'params': [ujson.dumps(params)],
                              'origin': package['origin'],
                              'input_image': ''}
            if (CONFIG['local']):
                print(format_package)
            # cal prices
            cost_points = 1
            # if package['mode'] == 'cert':
            #     params['batch_size'] = 1
            # if package['mode'] == 'hires':
            #     if params.get('hires_times'):
            #         if int(params['hires_times']) == 3:
            #             cost_points = 2
            #         elif int(params['hires_times']) == 4:
            #             cost_points = 3

                        # elif package['mode'] == 'avatar':
            #     batch_size = int(params['batch_size'])
            #
            #     if batch_size == 1:
            #         cost_points = 5
            #     elif batch_size == 2:
            #         cost_points = 8
            #
            # elif package['mode'] == 'mirror':
            #     batch_size = int(params['batch_size'])
            #
            #     if batch_size == 1:
            #         cost_points = 5
            #     elif batch_size == 2:
            #         cost_points = 8
            #
            # elif package['mode'] == 'hair':
            #     batch_size = int(params['batch_size'])
            #
            #     if batch_size == 1:
            #         cost_points = 5
            #     elif batch_size == 2:
            #         cost_points = 8
            #

            # else:
            #     batch_size = int(params['batch_size'])
            #     cost_points = batch_size
                # if batch_size == 1:
                #     cost_points = 5
                # elif batch_size == 2:
                #     cost_points = 8
                # else:
                #     cost_points = 10

            # check balance
            account = (await request.app.ctx.supabase_client.table("account").select("*").eq("id", user_id).execute()).data[0]
            buf_result = {'success': True, 'result': None, 'act': None, 'type': package['mode']}

            if account['vip_level'] == 3:
                cost_points = 0

            if cost_points <= account['balance']:
                # recv image
                if package['mode'] == 'hires':
                    parsed_url = urlparse(package['chosen_image'])
                    # 获取查询参数
                    query_params = parse_qs(parsed_url.query)

                    category = query_params['category'][0]
                    category = category if category else "model"
                    # if category == 'hair':
                    #     dir_storage_path = CONFIG['storage_dirpath']['user_hair_dir']
                    # elif category == 'mirror':
                    #     dir_storage_path = CONFIG['storage_dirpath']['user_mirror_dir']
                    # elif category == 'avatar':
                    #     dir_storage_path = CONFIG['storage_dirpath']['user_avatar_dir']
                    # else:
                    dir_storage_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category)
                    # dir_storage_path = CONFIG['storage_dirpath'][f'user_{category}_dir']
                    img_fp = os.path.join(dir_storage_path, query_params['uid'][0], query_params['imgpath'][0])
                    format_package['input_image'] = img_fp

                elif package['mode'] == 'upscaler':
                    parsed_url = urlparse('?'+package['params']['inputUrl'].split('?')[-1])
                    # 获取查询参数
                    query_params = parse_qs(parsed_url.query)

                    if 'category' in query_params.keys() and 'imgpath' in query_params.keys():
                        category = query_params['category'][0]
                        category = category if category else "model"
                        dir_storage_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category)
                        # dir_storage_path = CONFIG['storage_dirpath'][f'user_{category}_dir']
                        img_fp = os.path.join(dir_storage_path, query_params['uid'][0], query_params['imgpath'][0])
                        format_package['input_image'] = img_fp
                    else:
                        format_package['input_image'] = os.path.join(CONFIG['storage_dirpath']['user_upload'],
                                                                     f"{user_id}.png")
                        if package['mode'] == 'facer':
                            format_package['input_image_tgt'] = os.path.join(
                                CONFIG['storage_dirpath']['user_facer_upload'],
                                f"{user_id}.png")

                else:
                    format_package['input_image'] = os.path.join(CONFIG['storage_dirpath']['user_upload'], f"{user_id}.png")
                    if package['mode'] == 'facer':
                        format_package['input_image_tgt'] = os.path.join(CONFIG['storage_dirpath']['user_facer_upload'],
                                                                     f"{user_id}.png")

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
                                    buf_result['result'] = f"{queue_list.index(str(task_result)) + 1}"
                                    # buf_result['result'] = f"第{queue_list.index(str(task_result)) + 1}位"
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
                    data = await request.app.ctx.supabase_client.table("transaction").insert({"user_id": user_id,
                                                                                               'amount': cost_points,
                                                                                               'is_plus': False,
                                                                                               'status': 1,
                                                                                               }).execute()
                    res = (await request.app.ctx.supabase_client.table("account").update(
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


async def qinghua_genreate(request: Request, ws):
    try:
        logger.info(f"received generate request")
        user_id = request.args['user_id'][0]

        if user_id.replace('device_', '') not in auth_devices:
            task_result = {'success': False, 'result': "backend.api.error.device.auth"}

            await ws.send(ujson.dumps(task_result))

        else:
            try:
                # recv params
                raw_msg = await ws.recv()

                package = ujson.loads(raw_msg)
                params = package['params']
                format_package = {'mode': [package['mode']],
                                  'user_id': [package['user_id']],
                                  'params': [ujson.dumps(params)],
                                  'origin': request.headers.origin,
                                  'input_image': ''}

                if(CONFIG['local']):
                    print(format_package)

                buf_result = {'success': True, 'result': None, 'act': None, 'type': package['mode']}
                # recv image
                if package['mode'] == 'hires':
                    parsed_url = urlparse(package['chosen_image'])
                    # 获取查询参数
                    query_params = parse_qs(parsed_url.query)

                    category = query_params['category'][0]
                    category = category if category else "model"
                    dir_storage_path = os.path.join(CONFIG['storage_dirpath'][f'user_storage'], category)
                    # dir_storage_path = CONFIG['storage_dirpath'][f'user_{category}_dir']
                    img_fp = os.path.join(dir_storage_path, query_params['uid'][0], query_params['imgpath'][0])
                    format_package['input_image'] = img_fp
                else:
                    format_package['input_image'] = os.path.join(CONFIG['storage_dirpath']['user_upload'], f"{user_id}.png")
                    if package['mode'] == 'facer':
                        format_package['input_image_tgt'] = os.path.join(CONFIG['storage_dirpath']['user_facer_upload'],
                                                                     f"{user_id}.png")

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
                                    buf_result['result'] = f"{queue_list.index(str(task_result)) + 1}"
                                    # buf_result['result'] = f"第{queue_list.index(str(task_result)) + 1}位"
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
