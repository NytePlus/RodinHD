import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from rembg import remove, new_session
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def remove_background_floodFill(img_np, bg_color = None, fill_color = (0, 0, 0), no_torlerance = True):
    if bg_color is None:
        bg_color = img_np[0, 0].tolist()

    h, w = img_np.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    if no_torlerance:
        lo, hi = (0, 0, 0), (0, 0, 0)
    else:
        lo, hi = (10, 10, 10), (10, 10, 10)
    for (x, y) in [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]:
        if np.linalg.norm(np.array(img_np[y, x]) - np.array(bg_color), ord=1) <= 60:
            cv2.floodFill(img_np, mask, (x, y), fill_color, lo, hi, cv2.FLOODFILL_FIXED_RANGE)

    return img_np

base_dir = '../data/portrait3d_data'
out_dir = '../data/portrait3d_nobg'
num_threads = 1
dirs = os.listdir(base_dir)
os.makedirs(out_dir, exist_ok=True)
def process(i):
    session = new_session('birefnet-portrait')
    thread_dirs = dirs[i:][::num_threads]
    for dir in thread_dirs:
        os.makedirs(os.path.join(base_dir, dir).replace(base_dir.split('/')[-1], out_dir.split('/')[-1]), exist_ok=True)
        imgs = glob.glob(os.path.join(base_dir, dir, '*.png'))
        print(os.path.join(base_dir, dir))
        outdir = os.path.join(base_dir, dir).replace(base_dir.split('/')[-1], out_dir.split('/')[-1])
        if len(os.listdir(outdir)) == 300:
            print('skip')
            continue
        for img in tqdm(imgs):
            cv_img = cv2.imread(img)
            # remove_background_floodFill(cv_img, fill_color= [100, 98, 97])
            cv_img = remove(cv_img, bgcolor=(255, 255, 255, 255), session=session)
            cv2.imwrite(img.replace(base_dir.split('/')[-1], out_dir.split('/')[-1]), cv_img)

if num_threads > 1:
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, i) for i in range(num_threads)]
    result = [future.result() for future in futures]
else:
    process(0);
