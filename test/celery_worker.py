# coding=utf-8
# @Time : 2023/8/9 上午9:36
# @File : celery_worker.py
import redis
from celery import Task, Celery, shared_task
import time

celery_name = 'celery_worker1'
app = Celery(celery_name, broker='pyamqp://localhost:5672', backend='redis://localhost:6379/0')
def qq(*args, **kwargs):
    print(args)

class _AddTask(Task):
    a = 2
    name = 'task'
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=1, decode_responses=True)

    def before_start(self, task_id, args, kwargs):
        print(self.redis_client.get('a'))
        print(task_id)
        print("before_start!!!!!!!!!!!!!!!!!!!!")

    def run(self, x, y):
        print("self.task_id")
        # print(self.id)
        qq(self)
        print("11111")
        for i in range(20):
            self.update_state(state='PROGRESS', meta={'progress': i + 1})
            time.sleep(1)

        print(f"!!!!!!!!!!!!!x: {x} self.a:{self.a}!!!!!!!!!!!!!!!!!!")
        return x + y
add = app.register_task(_AddTask)

if __name__ == '__main__':
    print(app.tasks)
    # app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1'])
    app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1', '-P', 'solo', '-Ofair', '-n', 'w2'])
