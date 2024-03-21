import base64
import inspect
import time
import traceback
from importlib import import_module
from io import BytesIO

import redis
from lib.celery_workshop.wokrshop import WorkShop
from lib.common.common_util import logging
from lib.celery_workshop.operator import Operator
from utils.global_vars import CONFIG

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    running_workshop = []
    try:
        # clear queue
        redis_client = redis.StrictRedis(host='localhost', port=6379, db=1, decode_responses=True)
        # redis_client.delete('celery_task_queue')
        # redis_client.delete('celery_task_revoked')
        redis_client.flushdb()

        module = import_module(f'operators.sora_video')
        all_classes = inspect.getmembers(inspect.getmodule(module), inspect.isclass)
        for c_name, c_obj in all_classes:
            if c_obj is not Operator and issubclass(c_obj, Operator):
                workshop = WorkShop(c_obj)
        else:
            workshop.register()
            # clear queue
            workshop.clear_queue()

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
