import json
import sys


class Operator(object):
    def __init__(self):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def operation(self, *args, **kwargs):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def update_progress(self, celery_task, redis_client, p):
        # cur_state = celery_task.AsyncResult(celery_task.request.id).state
        # print(f"cur_state: {cur_state}")
        # revoked_list = redis_client.lrange('celery_task_revoked', 0, -1)
        if redis_client.get(celery_task.request.id):
            celery_task.update_state(state='REVOKED')
            # redis_client.lrem('celery_task_revoked', count=1, value=celery_task.request.id)
            redis_client.delete(celery_task.request.id)
            return True

        else:
            celery_task.update_state(state='PROGRESS', meta={'progress': p})
            return False

