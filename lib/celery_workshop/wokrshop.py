# coding=utf-8
# @Time : 2023/8/9 上午11:08
# @File : wokrshop.py
import os
import subprocess
import sys
from collections import OrderedDict
from lib.common.NoDaemonProcessPool import NoDaemonProcess
from celery import Celery
import GPUtil


class WorkShop(object):
    """Celery application.
    """
    proc = []

    def __init__(self, op):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")
        self.op = op

    @staticmethod
    def get_celery_app_name(index, op_name, is_cuda):
        return f"{op_name}_{'cuda' if is_cuda else 'cpu'}_{index}"

    # 指定独立存在的子进程，处理业务的进程，存储加载后的模型
    @staticmethod
    def instance_worker_proc(index, op_name, is_cuda):
        print("index, op_name, is_cuda")
        print(index, op_name, is_cuda)
        import traceback
        import os
        from importlib import import_module
        from celery import Task, Celery
        from lib.common.common_util import logging
        from utils.global_vars import CONFIG

        celery_app_name = WorkShop.get_celery_app_name(index, op_name, is_cuda)
        while True:
            try:
                app = Celery(celery_app_name, broker='amqp://localhost:5672', backend='redis://localhost:6379/0')

                module = import_module(f'operators')

                class ProceedTask(Task):
                    # load model
                    operator = getattr(module, op_name)(index)
                    # task 命名
                    name = f"{celery_app_name}.ProceedTask"

                    def run(self, *args, **kwargs):
                        res = self.operator(*args, **kwargs)
                        return res

                task = app.register_task(ProceedTask)
                print(app.tasks)

                # , '--pool=eventlet'
                app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1', '-P', 'solo'])
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
    def __call__(self, *args, **kwargs):
        # 获取显存占用最小的显卡idx
        cuda_device_idx = GPUtil.getAvailable(order='memory', limit=1)[0] if self.op.cuda and len(GPUtil.getGPUs()) > 1 else 0

        celery_app_name = self.get_celery_app_name(cuda_device_idx, self.op.__name__, self.op.cuda)
        app = Celery(celery_app_name, broker='amqp://localhost:5672', backend='redis://localhost:6379/0')
        target_task_name = f'{celery_app_name}.ProceedTask'
        task_result = app.send_task(target_task_name, args=args, kwargs=kwargs)
        return task_result

    # 服务端建立celery生产者进程
    def register(self):
        if self.op.cuda:
            for proc_idx in range(self.op.num):
                args = OrderedDict({'index': proc_idx,
                                    'op_name': self.op.__name__,
                                    'cuda': self.op.cuda,
                                    })
                # self.instance_worker_proc(*args.values())
                proc = NoDaemonProcess(target=self.instance_worker_proc,
                                                  args=tuple(i for i in args.values())).start()
                self.proc.append(proc)
                # self.proc = subprocess.Popen(f"python worker_proc.py {' '.join([str(i) for i in args.values()])}", shell=True)


        else:
            args = OrderedDict({'op_name': self.op.__name__,
                                'cuda': self.op.cuda,
                                })
            proc = NoDaemonProcess(target=self.procedure_worker_proc, args=tuple(i for i in args.values()))
            self.proc.append(proc)
            # self.proc = subprocess.Popen(f"python worker_proc.py {''.join([i for i in args.values()])}", shell=True)

