# coding=utf-8
# @Time : 2023/8/9 上午11:08
# @File : wokrshop.py
import json
import sys
from collections import OrderedDict
from lib.common.NoDaemonProcessPool import NoDaemonProcess
from celery import Task, Celery




class WorkShop(object):
    """Celery application.
    """
    proc = None

    def __init__(self, op):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")
        self.op = op
        self.celery_app_name_prefix = f"{self.op.__name__}_{'cuda' if self.op.cuda else 'cpu'}"

    # 指定独立存在的子进程，处理业务的进程，存储加载后的模型
    @staticmethod
    def instance_worker_proc(index, op_name, is_cuda):
        import traceback
        from importlib import import_module
        from celery import Task, Celery
        from lib.common.common_util import logging
        from utils.global_vars import CONFIG

        while True:
            try:
                app = Celery(f"{op_name}_{index}", broker='amqp://localhost:5672', backend='redis://localhost:6379/0')

                module = import_module(f'operators')

                class ProceedTask(Task):
                    operator = getattr(module, op_name)()

                    def run(self, *args, **kwargs):
                        res = self.operator.operation(*args, **kwargs)
                        return res

                task = app.register_task(ProceedTask)
                app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1'])
            except Exception:
                print(traceback.format_exc())
                logging(
                    f"{__file__} fatal error: {traceback.format_exc()}",
                    f"logs/error.log", print_msg=CONFIG['debug_mode'])

    # 用完就销毁的子进程，处理业务的进程
    @staticmethod
    def procedure_worker_proc(op_name, is_cuda):
        # TODO
        pass

    # 客户端异步调用发布任务，订阅任务结果
    async def __call__(self):
        # TODO
        cuda_device_idx = 0 if self.op.cuda else 0
        celery_app_name = f"{self.celery_app_name_prefix}_{cuda_device_idx}"
        app = Celery(celery_app_name, backend='redis://localhost:6379/0')
        task_result = await app.send_task(f'{celery_app_name}.ProceedTask', args=[1, 4])
        return task_result.result

    # 服务端建立celery生产者进程
    def register(self):
        if self.op.cuda:
            for proc_idx in range(self.op.num):
                args = OrderedDict({'index': proc_idx,
                                    'op_name': self.op.__name__,
                                    'cuda': self.op.cuda,
                                    })
                # self.instance_worker_proc(*args.values())
                self.proc = NoDaemonProcess(target=self.instance_worker_proc,
                                            args=tuple(i for i in args.values())).start()

        else:
            args = OrderedDict({'op_name': self.op.__name__,
                                'cuda': self.op.cuda,
                                })
            self.proc = NoDaemonProcess(target=self.procedure_worker_proc, args=tuple(i for i in args.values()))
