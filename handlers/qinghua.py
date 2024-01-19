import os.path

import traceback
import urllib.parse
from datetime import datetime, timedelta, date
import aiofile
import ujson
from sanic.response import json as sanic_json, file_stream, text
from sanic.views import HTTPMethodView
from lib.celery_workshop.wokrshop import WorkShop
from lib.common.common_util import encrypt, generate_random_digits, uuid_to_number_string

from gotrue import check_response
from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from operators import OperatorSD
from utils.global_vars import CONFIG
sd_workshop = WorkShop(OperatorSD)
temp_udb = [f'test_{"%03d" % i}' for i in range(1, 11)]


auth_devices = ['test']


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




