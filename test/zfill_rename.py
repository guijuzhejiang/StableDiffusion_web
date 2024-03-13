# coding=utf-8
# @Time : 2024/3/13 下午1:15
# @File : zfill_rename.py
import glob
import os

dir_path = '/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/gallery/*.webp'

for i in glob.glob(dir_path):
    os.rename(i, os.path.join(dir_path, f'{str(int(os.path.basename(i).replace(".webp", ""))).zfill(6)}.webp'))
