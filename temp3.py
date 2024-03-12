import glob
import sys
from datetime import datetime, timedelta
from collections import OrderedDict
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
            webp_path = os.path.join(folder_path, webp_filename)
            # 保存为 WebP 格式
            image.save(webp_path, 'WEBP')

# 调用函数，替换下面的 'your_folder_path' 为你的 PNG 文件所在的文件夹路径
# convert_png_to_webp('your_folder_path')


import pandas as pd

# 指定 Excel 文件路径
excel_file_path = '/home/ray/Workspace/file/SDXL实装.xlsx'

# 使用 pandas 读取 Excel 文件的特定行和列
# header=None 表示原始数据没有列名，sheet_name 参数根据你的实际情况调整
# usecols='A:E' 表示只读取 A 到 E 列，skiprows 跳过前 5 行（因为行索引从 0 开始，且不包括结束行），nrows 读取 102 行（因为不包括起始行）
df = pd.read_excel(excel_file_path, usecols='A:E', skiprows=range(5), nrows=102, header=None, sheet_name='SDXL风格')

# 遍历 DataFrame，并输出 A 列和 E 列的数据
# 在这里，列的索引是从 0 开始的，所以 A 列是 0，E 列是 4
ddd = OrderedDict()
count = 0
for index, row in df.iterrows():
    if row[4]:
        count = count + 1
        name = row[1].split('-')[-1]
        prompt = row[2]
        ddd[count] = {'label': name, 'prompt': prompt, 'disallow':[]}

print(dict(ddd))
# image = Image.open('/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/magic_text2image/Pokémon.png')
# image.save(os.path.join('/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/magic_text2image/Pokémon.webp'))
