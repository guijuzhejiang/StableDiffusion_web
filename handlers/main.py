from celery import Celery
from sanic.views import HTTPMethodView
from sanic.response import redirect

temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


class Test(HTTPMethodView):
    async def post(self, request):
        celery_name = 'OperatorOCR_cuda_0'
        # celery_name = 'celery_worker0'
        app0 = Celery(celery_name, broker='amqp://localhost:5672', backend='redis://localhost:6379/0')
        task_result = await app0.send_task(str(f'{celery_name}.ProceedTask'), args=[1, 4])
        return task_result.result
