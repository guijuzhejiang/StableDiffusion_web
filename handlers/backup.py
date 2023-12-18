# coding=utf-8
# @Time : 2023/12/15 下午2:20
# @File : backup.py

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