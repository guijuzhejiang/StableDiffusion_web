import base64
import time
import traceback
from io import BytesIO

import ujson

from lib.celery_workshop.util import load_workshops
from lib.common.common_util import logging
from lib.redis_mq import SyncRedisMQ
from utils.global_vars import CONFIG

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    running_workshop = []
    try:
        if CONFIG['debug_mode']:
            redis_mq = SyncRedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'])
            workers = {}
            for workshop in load_workshops():
                if workshop.op.enable:
                    workers[workshop.op.__name__] = workshop.op()

            for msg_id, msg in redis_mq.consume():
                # print(msg)
                logging(
                    f"{__file__}got msg: {msg}",
                    f"logs/info.log", print_msg=CONFIG['debug_mode'])
                params = ujson.loads(msg['params'])
                params['input_image'] = [BytesIO(base64.b64decode(msg['input_image']))]
                res = workers['OperatorSD'](**params)
                # res = {'success': False, 'result':"fatal error"}
                redis_mq.pub(msg['reply_queue_name'], {'res': ujson.dumps(res)}, CONFIG['server']['msg_expire_secs'])
        else:
            # 从operators加载服务对像并开启进程和celery worker
            for workshop in load_workshops():
                if workshop.op.enable:
                    # init workers && load models
                    workshop.register()
                    running_workshop.append(workshop)

        while True:
            time.sleep(5)

    except Exception:
        print(traceback.format_exc())
        logging(
            f"{__file__} fatal error: {traceback.format_exc()}",
            f"logs/error.log", print_msg=CONFIG['debug_mode'])
    finally:
        for w in running_workshop:
            for p in w.proc:
                p.terminate()
