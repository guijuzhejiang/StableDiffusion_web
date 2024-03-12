import glob
import sys
from datetime import datetime, timedelta
from collections import OrderedDict

import aiosupabase
from PIL import Image
import os


from PIL import Image
import os

def convert_png_to_webp(folder_path):
    # 遍历指定文件夹内的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # 检查文件扩展名是否为 .png
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 打开 PNG 图片
            image = Image.open(file_path)
            # 构造 WebP 的文件名
            webp_filename = filename[:-4] + '.webp'
            webp_path = os.path.join('/home/ray/Workspace/file/素材/text_to_image/效果图webp/', webp_filename)
            # 保存为 WebP 格式
            image.save(webp_path, 'WEBP')

# 调用函数，替换下面的 'your_folder_path' 为你的 PNG 文件所在的文件夹路径
convert_png_to_webp('/home/ray/Workspace/file/素材/text_to_image/效果图/')


# import pandas as pd
#
# # 指定 Excel 文件路径
# excel_file_path = '/home/ray/Workspace/file/SDXL实装.xlsx'
#
# # 使用 pandas 读取 Excel 文件的特定行和列
# # header=None 表示原始数据没有列名，sheet_name 参数根据你的实际情况调整
# # usecols='A:E' 表示只读取 A 到 E 列，skiprows 跳过前 5 行（因为行索引从 0 开始，且不包括结束行），nrows 读取 102 行（因为不包括起始行）
# df = pd.read_excel(excel_file_path, usecols='B:D', skiprows=range(1), nrows=43, header=None, sheet_name='文生图sample')
#
# # 遍历 DataFrame，并输出 A 列和 E 列的数据
# # 在这里，列的索引是从 0 开始的，所以 A 列是 0，E 列是 4
# ddd = OrderedDict()
# count = 0
#
# user_id = 'd8f5d02e-5e36-4040-be84-15a0a2cf90e8'
# url = "https://www.guijutech.com:8888/"
# key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogInNlcnZpY2Vfcm9sZSIsCiAgImlzcyI6ICJzdXBhYmFzZSIsCiAgImlhdCI6IDE2OTU4MzA0MDAsCiAgImV4cCI6IDE4NTM2ODMyMDAKfQ.QanqKpEYyjqgvl1ElcWw7JJAvUEzIC0e0w1pFfPOITE"
# # supabase_client = create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'])
# supabase_client = aiosupabase.Supabase
# supabase_client.configure(
#     url=url,
#     key=key,
#     debug_enabled=True,
# )
#
# for index, row in df.iterrows():
#     img = Image.open(row[1])
#     w, h = img.size
#     config = {'width': w, 'height': h, 'style': int(row[2])}
#     supabase_client.table("gallery").insert({"config": config, 'prompt':row[3],'instance_id': os.path.basename(row[1].replace('png', 'webp'))}).execute()
# print(dict(ddd))
# image = Image.open('/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/magic_text2image/Pokémon.png')
# image.save(os.path.join('/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/magic_text2image/Pokémon.webp'))