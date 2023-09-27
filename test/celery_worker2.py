# coding=utf-8
# @Time : 2023/8/9 上午9:36
# @File : celery_worker.py
from celery import Task, Celery, shared_task
import time

celery_name = 'celery_worker0'
app = Celery(celery_name, broker='amqp://localhost:5672', backend='redis://localhost:6379/0')


class _AddTask2(Task):
    a = 123
    name = 'task'
    # bind = True
    def run(self, x, y):
        print(self.a)
        print("222222")
        self.update_state(state='PROGRESS', meta={'progress': 1})
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': 2})
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': 3})
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': 4})
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': 5})
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': 6})
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': 7})
        time.sleep(1)

        print(f"!!!!!!!!!!!!!x: {x} self.a:{self.a}!!!!!!!!!!!!!!!!!!")
        return x + y
print(_AddTask2.a)

app.register_task(_AddTask2)

if __name__ == '__main__':
    print(app.tasks)
    # app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1'])
    app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1', '-Ofair', '-n', 'w2'])
