import datetime
import glob
import os
import shutil

import yaml

YML_PATH = '/home/ray/Workspace/project/ocr/github/PaddleOCR/configs/rec/ch_PP-OCRv2/JP_PP-OCRv2_rec_distillation_finetuning.yml'
TARGET_PATH = '/home/ray/Workspace/project/ocr/rec'

INPUT_LIST = [
   i for i in glob.glob('/home/ray/Workspace/project/ocr/det_qinghua_all/*') if os.path.isdir(i) and datetime.datetime.fromtimestamp(os.path.getctime(i)) > datetime.datetime.now() - datetime.timedelta(days=2)
]

label_file_list = yaml.load(open(YML_PATH, 'rb'))['Train']['dataset']['label_file_list']
for input_dp in INPUT_LIST:
    data_name = os.path.basename(input_dp)
    target_dp = os.path.join(TARGET_PATH, data_name)
    target_lb_fp = os.path.join(TARGET_PATH, f"rec_gt_{data_name}.txt")

    if os.path.exists(target_dp):
        print(f"{target_dp} existed")
    else:
        shutil.copytree(os.path.join(input_dp, 'crop_img'), target_dp)
        shutil.copyfile(os.path.join(input_dp, 'rec_gt.txt'), target_lb_fp)
        # 复制数据到目标地址
        lb_content = None
        with open(target_lb_fp, 'r') as lb_fp:
            lb_content = lb_fp.readlines()

        new_content = [i.replace('crop_img', data_name) for i in lb_content]
        with open(target_lb_fp, 'w') as lb_fp:
            lb_fp.writelines(new_content)

        # 修改YML
        yml_content = None
        with open(YML_PATH, 'r') as yml_f:
            yml_content = yml_f.readlines()

        train_flag = False
        for index, row in enumerate(yml_content):
            if 'Train' in row:
                train_flag = True
            if 'label_file_list: ' in row and train_flag and '#' not in row:
                yml_content.insert(index+1, f"    - {target_lb_fp}\n")
                break
        for index, row in enumerate(yml_content):
            if 'ratio_list: ' in row:
                yml_content[index] = f"{row.split('[')[0]}[1,{row.split('[')[1]}"
                break
        with open(YML_PATH, 'w') as yml_f:
            yml_f.writelines(yml_content)
