import os
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

base_dir = '../data/portrait3d_data'
num_threads = 40
demage_imgs, demage_metadata = [], []
dirs = os.listdir(base_dir)
def process(i):
    thread_dirs = dirs[i:][::num_threads]
    for dir in thread_dirs:
        imgs = glob.glob(os.path.join(base_dir, dir, '*.png'))
        for img in imgs:
            cv_img = cv2.imread(img)
            if cv_img is None:
                demage_imgs.append(img)
            pbar.update(1)
        metadata = glob.glob(os.path.join(base_dir, dir, '*.json'))
        for data in metadata:
            try:
                with open(data, 'r') as file:
                    json.load(file)
            except:
                demage_metadata.append(data)
            pbar.update(1)

total_files = sum(len(glob.glob(os.path.join(base_dir, dir, '*.*'))) for dir in dirs)
with tqdm(total=total_files) as pbar:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, i) for i in range(num_threads)]
        result = [future.result() for future in futures]
if len(demage_imgs) == 0 and len(demage_metadata) == 0:
    print('data is safe.')
else:
    print(f'demage images: {demage_imgs}\n demage metadata: {demage_metadata}')