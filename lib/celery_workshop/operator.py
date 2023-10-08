import json
import sys


class Operator(object):
    def __init__(self):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def operation(self, *args, **kwargs):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def update_progress(self, celery_task, p):
        cur_state = celery_task.AsyncResult(celery_task.request.id).state
        print(f"cur_state: {cur_state}")
        if cur_state == 'REVOKED':
            celery_task.update_state(state='REVOKED')
            return True
        else:
            celery_task.update_state(state='PROGRESS', meta={'progress': p})

            return False

