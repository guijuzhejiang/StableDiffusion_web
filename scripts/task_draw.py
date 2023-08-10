# coding=utf-8
# @CREATE_TIME: 2021/8/20 上午9:14
# @LAST_MODIFIED: 2021/8/20 上午9:14
# @FILE: task_draw.py
# @AUTHOR: Ray
# -*- coding:utf-8 -*-
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import json
import os
import sys

sys.path.insert(0, ".")
import argparse
from lib.ocr.tools.infer.utility import draw_server_result
from paddlehub.common.logger import logger
import cv2

if __name__ == '__main__':
    # params for prediction engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='/home/ray/Workspace/project/ocr/github/gso_server/data/7001_002/upload/atGFmCSRyxNRTRNwQjiq.2021-08-20_10:31:36.7001_002.jpg')
    parser.add_argument("--json_result", type=str, default='/home/ray/Workspace/project/ocr/github/gso_server/data/7001_002/json/atGFmCSRyxNRTRNwQjiq.2021-08-20_10:31:36.7001_002.jpg.json')
    parser.add_argument("--inference_results_dp", type=str, default='/opt')
    parser.add_argument("--server_results_dp", type=str, default='/opt')
    parser.add_argument("--json_results_dp", type=str, default='/opt')
    params = parser.parse_args()

    # get params
    image_path = params.image_path
    server_results_dp = params.server_results_dp
    inference_results_dp = params.inference_results_dp
    json_results_dp = params.json_results_dp

    # get predict result
    fn = os.path.basename(image_path)
    with open(params.json_result, 'r') as f:
        res = json.load(f)

    # draw img style1
    draw_img = draw_server_result(image_path, res[0], logger)
    if draw_img is not None:
        if not os.path.exists(server_results_dp):
            os.makedirs(server_results_dp)
        cv2.imwrite(
            os.path.join(server_results_dp, fn),
            draw_img[:, :, ::-1])

    # draw img style2
    # image = Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    # boxes = [np.array([j for j in i['text_region']], dtype=np.float32) for i in res[0]]
    # txts = [i['text'] for i in res[0]]
    # scores = [i['confidence'] for i in res[0]]
    #
    # draw_img = draw_ocr_box_txt(
    #     image,
    #     boxes,
    #     txts,
    #     scores,
    #     drop_score=0.5,
    #     font_path='./doc/fonts/simfang.ttf')
    # if not os.path.exists(inference_results_dp):
    #     os.makedirs(inference_results_dp)
    # cv2.imwrite(
    #     os.path.join(inference_results_dp, fn),
    #     draw_img[:, :, ::-1])
    #
    # # save json
    # with open(os.path.join(json_results_dp, fn+'.json'), 'w') as f:
    #     json.dump(res, f)
