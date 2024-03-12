from PIL import Image
import os

def convert_png_to_webp(folder_path):
    # 遍历指定文件夹内的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # 检查文件扩展名是否为 .png
            # 构造 WebP 的文件名
            webp_filename = filename[:-4] + '.webp'
            webp_dir = os.path.join(folder_path, 'webp')
            os.makedirs(webp_dir, exist_ok=True)
            webp_path = os.path.join(webp_dir, webp_filename)
            if not os.path.exists(webp_path):
                # 构造完整的文件路径
                file_path = os.path.join(folder_path, filename)
                # 打开 PNG 图片
                image = Image.open(file_path)
                # 保存为 WebP 格式
                image.save(webp_path, 'WEBP')

# 调用函数，替换下面的 'your_folder_path' 为你的 PNG 文件所在的文件夹路径
convert_png_to_webp('/home/zzg/data/CV/text_to_image/效果图/')