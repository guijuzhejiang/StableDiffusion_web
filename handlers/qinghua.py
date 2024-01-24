import traceback
from sanic.response import json as sanic_json
from sanic.views import HTTPMethodView
from lib.celery_workshop.wokrshop import WorkShop
from operators import OperatorSD
sd_workshop = WorkShop(OperatorSD)
temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


auth_devices = ['866503050126727']


class DeviceAuthVerify(HTTPMethodView):
    """
        验证设备号
    """
    async def post(self, request):
        try:
            device_code = request.form['devicecode'][0]
            return sanic_json({'success': device_code in auth_devices, 'user': {'id': f'device_{device_code}'}})
        except Exception:
            print(traceback.format_exc())
            return sanic_json({'success': False, 'result': 'backend.api.error.device.auth'})




