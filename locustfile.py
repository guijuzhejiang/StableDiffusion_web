import datetime

from locust import HttpUser, task, between

from lib.common.common_util import generate_random


class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_blog(self):
        # with open('/home/ray/Workspace/project/ocr/test/915162328.jpg', 'rb') as f:
        #     img_data = f.read()
        header = {"Content-Type": "mutipart/form-data"}
        payload = {
            # "img_data": img_data,
            "file_type": 'jpg',
            "sid": generate_random(20),
            "bool_draw_img": 'false',
            "start_record_time": datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S'),
        }
        files = {'img_data': open('/home/ray/Workspace/project/ocr/test/915162328.jpg', 'rb')}
        req = self.client.post("http://www.guijubar.com:5003/upload", data=payload, files=files)
        if req.status_code == 200:
            print("success")
        else:
            print("fails")

        if 'error' in str(req.content):
            print("error!!!!!!!!")