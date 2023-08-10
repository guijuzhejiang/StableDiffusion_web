import time
import traceback

from lib.celery_workshop.util import load_workshops
from lib.common.common_util import logging
from utils.global_vars import CONFIG

if __name__ == '__main__':
    try:
        # 从operators加载服务对像并开启进程和celery worker
        for workshop in load_workshops():
            # init workers && load models
            workshop.register()

        while True:
            time.sleep(5)

    except Exception:
        print(traceback.format_exc())
        logging(
            f"{__file__} fatal error: {traceback.format_exc()}",
            f"logs/error.log", print_msg=CONFIG['debug_mode'])
