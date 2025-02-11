import os
import re
import cv2
import json
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm

source_dir = '/home/wcc/RodinHD/data/portrait3dexp_data'
target_dir = '/home/wcc/RodinHD/data/portrait3dexpclip_data'

min_elevation = 160
max_elevation = 180

def wh_adj(box, W, H):
    x, y, w, h = box
    size = max(w, h)

    new_x = x + (w - size) // 2
    new_y = y + (h - size) // 2

    new_x = max(0, new_x)
    new_y = max(0, new_y)

    if new_x + size > W:
        size = W - new_x
    if new_y + size > H:
        size = H - new_y
    return new_x, new_y, size, size

def pad(box, padding = 0.3):
    x, y, w, h = box
    pad_w = w * padding
    pad_h = h * padding

    new_x = max(x - pad_w, 0)
    new_y = max(y - pad_h, 0)

    new_w = w + pad_w * 2
    new_h = h + pad_h * 2

    return new_x, new_y, new_w, new_h

def wh2xy(box):
    return box[0], box[1], box[0] + box[2], box[1] + box[3]

if __name__ == '__main__':
    os.makedirs(target_dir, exist_ok=True)

    for dir_name in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, dir_name)
        target_subdir = os.path.join(target_dir, dir_name)
        if os.path.exists(target_subdir) or not os.path.isdir(subdir_path):
            continue
        os.makedirs(target_subdir, exist_ok=False)

        idx = 0
        front_img_path = re.sub(r'_[^\x00-\xff]+', '', os.path.join(subdir_path.replace('exp', ''), 'img_proc_fg_000000.png'))
        image = cv2.imread(front_img_path, cv2.IMREAD_UNCHANGED)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(300, 300))
        if len(faces) <= 0:
            print(f'Front image in {subdir_path} cannot detect face.')
            continue
        face_box = pad(wh_adj(faces[0], 1024, 1024), 0.4)

        subdir_path_list = sorted(os.listdir(subdir_path))
        for file_name in tqdm(subdir_path_list):
            if file_name == '纠结':
                print(subdir_path)
            if file_name.endswith('.json'):
                json_path = os.path.join(subdir_path, file_name)
                png_name = file_name.replace('metadata', 'img_proc_fg').replace('.json', '.png')
                png_path = os.path.join(subdir_path, png_name)

                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                camera_matrix = np.array(metadata['cameras'][0]['transformation'])

                up_vector = camera_matrix[:3, 1]

                elevation = np.arccos(up_vector[1]) * 180 / np.pi

                if min_elevation <= elevation <= max_elevation:
                    tgt_png_name = f'img_proc_fg_{idx:06d}.png'
                    tgt_json_name = f'metadata_{idx:06d}.json'
                    idx += 1

                    target_json_path = os.path.join(target_subdir, tgt_json_name)
                    shutil.copy2(json_path, target_json_path)

                    target_png_path = os.path.join(target_subdir, tgt_png_name)
                    img = Image.open(png_path)

                    img = img.crop(wh2xy(face_box))

                    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

                    img.save(target_png_path)

    print("处理完成！")