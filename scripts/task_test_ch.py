# coding=utf-8
# @CREATE_TIME: 2021/8/20 上午9:24
# @LAST_MODIFIED: 2021/8/20 上午9:24
# @FILE: task_test.py
# @AUTHOR: Ray
import datetime
import glob
import os
import sys

from PIL import Image

from lib.ocr.deploy.hubserving.ocr_system.module import OCRSystem
from lib.ocr.tools.infer.predict_system import TextSystem

sys.path.insert(0, ".")
from lib.ocr.tools.infer.utility import draw_ocr_box_txt
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    #
    input_dir = "/home/ray/Workspace/project/ocr/test/ch/ID0330"
    output_dir = "/home/ray/Workspace/project/ocr/test/ch/out"

    #
    model_list = [
        '/home/ray/Workspace/models/ocr/rec/zh_distillation_v2.4_finetuning_20220403',
        # '/home/ray/Workspace/project/ocr/github/PaddleOCR_GJ/output/rec/japan_mobileV3_small_pretrain_20210903/infer_latest'
                  ]
    model_name = [
        'zh_distillation_v2.4_finetuning_20220403',
        # 'db_mv3_finetuning_20211209_det'
                  ]

    ocr = OCRSystem(use_gpu=True, auto_load=False)
    ocr.cfg.det_model_dir = '/home/ray/Workspace/models/ocr/det/ch_PP-OCRv2_det_infer'
    ocr.cfg.rec_char_dict_path = '/home/ray/Workspace/project/ocr/src/gso_server_ch_20220330/lib/ocr/ppocr/utils/dict/ppocr_keys_20220401.txt'
    # ocr.cfg.det_model_dir = '/home/ray/Workspace/project/ocr/github/PaddleOCR/output/det/JP/db_mv3_finetuning_20211209/infer_epoch100/Teacher'
    ocr.cfg.det_limit_side_len = 960
    ocr.cfg.use_mp = True
    ocr.cfg.use_multiprocess = True
    ocr.cfg.batch = True

    total_time = datetime.datetime.now()
    for model_index, model in enumerate(model_list):
        for det_limit_side_len in [960 + (2560-960) * i for i in range(2)]:
            ocr.cfg.det_limit_side_len = det_limit_side_len
            for det_db_unclip_ratio in [1.6 + 0.1*i for i in range(1)]:
                ocr.cfg.rec_model_dir = model
                ocr.cfg.det_db_unclip_ratio = det_db_unclip_ratio
                ocr.text_sys = TextSystem(ocr.cfg)

                for image_path in sorted(glob.glob(f"{input_dir}/*")):
                    # predict
                    start_time = datetime.datetime.now()
                    try:
                        res = ocr.predict(paths=[image_path])
                    except Exception:
                        # worksheet.insert_image(cur_row, cur_col, 'error')
                        continue
                    predict_time = datetime.datetime.now() - start_time
                    print(f"!!!!!!!  {datetime.datetime.now() - start_time}  !!!!!!!!")

                    # draw img style2
                    start_time = datetime.datetime.now()
                    image = Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
                    boxes = [np.array([j for j in i['text_region']], dtype=np.float32) for i in res[0]]
                    txts = [i['text'] for i in res[0]]
                    scores = [i['confidence'] for i in res[0]]

                    draw_img = draw_ocr_box_txt(
                        image,
                        boxes,
                        txts,
                        scores,
                        drop_score=0.5,
                        font_path='./doc/fonts/simfang.ttf')

                    output_dp = os.path.join(output_dir, model_name[model_index], str(det_limit_side_len), str(det_db_unclip_ratio))
                    if not os.path.exists(output_dp):
                        os.makedirs(output_dp)

                    img_fp = os.path.join(output_dp, f'{str(det_limit_side_len)}_{str(det_db_unclip_ratio)}_{os.path.basename(image_path)}')

                    cv2.imwrite(img_fp, draw_img[:, :, ::-1])

                    draw_time = datetime.datetime.now() - start_time

    print(f"~~~~~~~~~~~{datetime.datetime.now() - total_time}~~~~~~~~~~~")
    # get result
    # with open(res_fp, 'w') as f:
    #     f.writelines(csv_res)
    # workbook.close()
