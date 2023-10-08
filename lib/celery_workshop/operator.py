import json
import sys


class Operator(object):
    def __init__(self):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def operation(self, *args, **kwargs):
        print(f"run {self.__class__.__name__}:{sys._getframe().f_code.co_name}")

    def update_progress(self, celery_task, p):
        if celery_task.AsyncResult(celery_task.request.id).state == 'REVOKED':
            celery_task.update_state(state='REVOKED')
            return True
        else:
            celery_task.update_state(state='PROGRESS', meta={'progress': p})

            return False

