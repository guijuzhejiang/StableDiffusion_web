# coding=utf-8
# @Time : 2023/8/9 上午9:51
# @File : celery_test.py
from celery import Celery
if __name__ == "__main__":
    celery_name = 'celery_worker0'
    # celery_name = 'celery_worker0'
    app0 = Celery(celery_name, broker='amqp://localhost:5672', backend='redis://localhost:6379/0')

    print(app0.control.inspect().stats())
    task_result = app0.send_task(str('task'), args=[1, 4])
    print(app0.control.inspect().stats())

    # task_result = app0.send_task(str(f'{celery_name}._AddTask'), args=[1, 4])
    result = task_result.get()
    print(result)
