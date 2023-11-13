import json
import sys

import redis


class Operator(object):
    def __init__(self):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=1, decode_responses=True)

    def __call__(self, *args, **kwargs):
        # get celery task
        self.celery_task = args[0]

    def update_progress(self, p):
        # cur_state = celery_task.AsyncResult(celery_task.request.id).state
        # print(f"cur_state: {cur_state}")
        # revoked_list = redis_client.lrange('celery_task_revoked', 0, -1)
        if self.redis_client.get(self.celery_task.request.id):
            self.celery_task.update_state(state='REVOKED')
            # self.redis_client.lrem('celery_task_revoked', count=1, value=celery_task.request.id)
            self.redis_client.delete(self.celery_task.request.id)
            return True

        else:
            self.celery_task.update_state(state='PROGRESS', meta={'progress': p})
            return False

