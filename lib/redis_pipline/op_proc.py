import datetime
import sys
import time
import traceback
from importlib import import_module

from lib.common.NoDaemonProcessPool import NoDaemonProcess
from lib.common.common_util import logging
from lib.redis_mq import SyncRedisMQ
from utils.global_vars import CONFIG

redis_mq = SyncRedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'])


class WorkerProc:
    def __init__(self, size, operator_name=None):
        self.proc_pool = [None for x in range(size)]
        self.proc_pool_size = size
        self.operator_name = operator_name

        if size == 1:
            module = import_module(f'operators')
            self.operator = getattr(module, operator_name)()

    def __call__(self, message, send_queue_name):
        result = False

        if self.proc_pool_size == 1:
            result = self.operator.operation(message)

        else:
            # 如果有空余位置添加任务
            for i, t in enumerate(self.proc_pool):
                if t is None:
                    temp_task = NoDaemonProcess(target=self.__multi_worker_proc,
                                                args=(message, self.operator_name, send_queue_name))
                    temp_task.start()
                    self.proc_pool[i] = temp_task
                    break
            else:
                # 等待某个worker完成
                while True:
                    stop_flag = False
                    for i, t in enumerate(self.proc_pool):
                        if t.exitcode is not None:
                            t.close()
                            temp_task = NoDaemonProcess(target=self.__multi_worker_proc,
                                                        args=(message, self.operator_name, send_queue_name))
                            temp_task.start()
                            self.proc_pool[i] = temp_task
                            stop_flag = True
                            break
                    if stop_flag:
                        break
                    else:
                        time.sleep(0.2)

        return result

    @staticmethod
    def __multi_worker_proc(_msg, _op_name, _pub_queue_name):
        # from importlib import import_module
        # from utils.global_vars import CONFIG
        # from lib.redis_pipline.redis_mq import SyncRedisMQ
        # redis_mq = SyncRedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'])

        module = import_module(f'operators')
        op = getattr(module, _op_name)
        operator = op()

        try:
            res = operator.operation(_msg)
        except Exception:
            _msg['error'] = f"error in {__name__}:{sys._getframe().f_code.co_name}"
            redis_mq.pub(_msg['reply_queue_name'], _msg, CONFIG['server']['msg_expire_secs'])
        else:
            if 'error' in res.keys():
                redis_mq.pub(_msg['reply_queue_name'], res, expire_secs=CONFIG['server']['msg_expire_secs'])
            else:
                redis_mq.pub(_pub_queue_name, res,
                             expire_secs=CONFIG['server']['msg_expire_secs'] if _pub_queue_name == _msg[
                                 'reply_queue_name'] else 0)

    @staticmethod
    def worker_main_proc(worker_num, op_name, pipe_in_queue_name):
        worker_num = int(worker_num)

        # worker
        worker = WorkerProc(worker_num, op_name)
        logging(f"[{datetime.datetime.now()}]{worker.operator_name} init worker succeed",
                f"logs/info.log", print_msg=CONFIG['debug_mode'])

        # redis
        while True:
            try:
                logging(
                    f"[{datetime.datetime.now()}]{worker.operator_name} start consuming - pipe name :{pipe_in_queue_name}",
                    f"logs/info.log", print_msg=CONFIG['debug_mode'])
                # start consuming
                for msg_id, msg in redis_mq.consume(pipe_in_queue_name):
                    try:
                        logging(
                            f"{worker.operator_name} got msg: {msg}",
                            f"logs/info.log", print_msg=CONFIG['debug_mode'])
                        op_list = msg['pipeline'].split(',')
                        pub_queue_name = msg['reply_queue_name'] if op_list[-1] == pipe_in_queue_name else op_list[
                            op_list.index(pipe_in_queue_name) + 1]

                        res = worker(msg, pub_queue_name)
                    except Exception:
                        msg['error'] = f"error in {__name__}:{sys._getframe().f_code.co_name}"
                        redis_mq.pub(msg['reply_queue_name'], msg, CONFIG['server']['msg_expire_secs'])
                    else:
                        if worker.proc_pool_size == 1:
                            if 'error' in res.keys():
                                redis_mq.pub(msg['reply_queue_name'], res,
                                             expire_secs=CONFIG['server']['msg_expire_secs'])
                            else:
                                redis_mq.pub(pub_queue_name, res,
                                             expire_secs=CONFIG['server']['msg_expire_secs'] if pub_queue_name == msg[
                                                 'reply_queue_name'] else 0)
            except Exception:
                logging(
                    f"{__file__}|{sys.argv} fatal error: {traceback.format_exc()}",
                    f"logs/error.log", print_msg=CONFIG['debug_mode'])


if __name__ == '__main__':
    # argv : [op, consume_queue_name, pub_queue_name]
    _, _worker_num, _op_name, _pipe_in_queue_name = sys.argv
    WorkerProc.worker_main_proc(_worker_num, _op_name, _pipe_in_queue_name)

