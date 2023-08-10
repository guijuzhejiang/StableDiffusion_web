# -*- coding:utf-8 -*-
import argparse
import datetime
import os

start_time = datetime.datetime.now()
from lib.ocr.deploy.hubserving.ocr_system.module import OCRSystem
from lib.common.singleton_config import config

# params for prediction engine
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default=config["ocr"]["lang"])
parser.add_argument("--use_gpu", type=bool, default=config["ocr"]["use_gpu"])
parser.add_argument("--use_tensorrt", type=bool, default=config["ocr"]["use_tensorrt"])
parser.add_argument("--mq_name", type=str, default=config["mq_name"])
parser.add_argument("--img_path", type=str, default='/home/ubuntu/test_ratio2.jpg')
params = parser.parse_args()
    
if __name__ == "__main__":
    if params.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if 'LD_LIBRARY_PATH' not in os.environ:
            os.environ['LD_LIBRARY_PATH'] = config['env']['LD_LIBRARY_PATH']

        else:
            os.environ['LD_LIBRARY_PATH'] += ":" + config['env']['LD_LIBRARY_PATH']

    # load model
    ocr = OCRSystem(use_gpu=params.use_gpu, use_tensorrt=params.use_tensorrt)

    # 图片路径
    img_path = params.img_path
    print(img_path)
    if os.path.exists(img_path):
        # predict
        for i in range(4):
            start_time = datetime.datetime.now()
            res = ocr.predict(paths=[img_path])
            time_point_end = datetime.datetime.now()
            print(time_point_end-start_time)

