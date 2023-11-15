# coding=utf-8
# @Time : 2023/8/9 上午9:51
# @File : celery_test.py
import time

import redis
import requests
import ujson
from celery import Celery
from pyrabbit.api import Client
from pyrabbit.http import HTTPError


def on_raw_message(body):
    print("body!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(body)


def get_queue_number():
    count = '...'

    try:
        queue_name = 'celery'
        url = "http://localhost:15672/api/queues/%2F/celery"  # 替换为实际的URL
        username = 'guest'
        password = 'guest'

        # 构建认证的HTTP请求
        response = requests.get(url, auth=(username, password), json={"count": 999999, "ackmode": "ack_requeue_true", "encoding": "auto"}, headers={'Content-Type': 'application/json'})
        queue_info = response.json()
        print(queue_info)
    except HTTPError as e:
        print("Exception: Could not establish to rabbitmq http api: " + str(
            e) + " Check for port, proxy, username/pass configuration errors")
        raise

    return count

if __name__ == "__main__":
    celery_name = 'celery_worker0'
    # celery_name = 'celery_worker0'
    app0 = Celery(celery_name, broker='amqp://localhost:5672', backend='redis://localhost:6379/0',task_track_started=True)
    # app0.conf.update(task_track_started=True)
    print(app0.control.inspect().reserved())
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=1)
    redis_client.set('a', 'b')
    task_result = app0.send_task(str('task'), args=[1, 1])
    task_result2 = app0.send_task(str('task'), args=[2, 2])
    task_result3 = app0.send_task(str('task'), args=[3, 3])
    task_result5 = app0.send_task(str('task'), args=[5, 5])

    task_result4 = app0.send_task(str('task'), args=[4, 4])
    # queue_list = app0.control.inspect().reserved()['celery@w2']
    # for index, q in enumerate(queue_list):
    #     if str(task_result4) == q['id']:
    #         print(f'{index+1}/{len(queue_list)}')
    # print(app0.control.inspect().scheduled())
    while task_result4.state != 'SUCCESS':
        # queue_list = task_result4.app.control.inspect().reserved()['celery@w2']

        print(get_queue_number())
        # for index, q in enumerate(queue_list):
        #     if str(task_result4) == q['id']:
        #         print(f'{index + 1}/{len(queue_list)}')
        # print(task_result4.state)
        time.sleep(0.1)
    # result = task_result.get(on_message=on_raw_message, propagate=False)
    # time.sleep(1)
    # print(task_result.result)
