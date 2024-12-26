import os
from PIL import Image

def convert_rgb_to_rgba(image_path, save_path):
    # 打开图像
    with Image.open(image_path) as img:
        # 如果是RGB图像，将其转换为RGBA
        if img.mode == 'RGB':
            rgba_img = img.convert('RGBA')
            # 保存为4通道的PNG图像
            rgba_img.save(save_path)
            print(f"Converted {image_path} to {save_path}")
        else:
            print(f"Skipped {image_path} (not RGB)")

def convert_images_in_directory(directory_path):
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 只处理PNG文件
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                save_path = os.path.join(root, f"converted_{file}")
                convert_rgb_to_rgba(file_path, save_path)

# 指定要处理的目录
if __name__ == "__main__":
    directory_path = './'  # 替换为你目录的路径
    convert_images_in_directory(directory_path)
