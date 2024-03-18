# coding=utf-8
# @Time : 2024/3/13 下午1:15
# @File : zfill_rename.py
import glob
import os

dir_path = '/home/zzg/workspace/webstorm/guiju_dashboard/public/assets/gallery/*.webp'

for i in glob.glob(dir_path):
    dir = os.path.dirname(i)
    pic_name = os.path.basename(i)
    if len(pic_name.split('.')[0]) != 6:
        new_path = os.path.join(dir, f'{str(int(pic_name.replace(".webp", ""))).zfill(6)}.webp')
        os.rename(i, new_path)
        print(f'{new_path} is renamed.')