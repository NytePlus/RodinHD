import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def read_keypoints(file_path):
    """
    读取关键点文件，返回 x, y, z 坐标的数组
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        x = list(map(float, lines[0].strip().split()))  # 读取 x 坐标
        y = list(map(float, lines[1].strip().split()))  # 读取 y 坐标
        z = list(map(float, lines[2].strip().split()))  # 读取 z 坐标
    return x, y, z

if __name__ == '__main__':
    x1, y1, z1 = read_keypoints('/home/wcc/3DDFA/samples/1-0_0.txt')
    x2, y2, z2 = read_keypoints('/home/wcc/3DDFA/samples/1-1_0.txt')

    image = cv2.imread('/home/wcc/3DDFA/samples/1-0.png')
    if image is None:
        print("图片读取失败，请检查路径是否正确。")
    else:
        image = np.array(image)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        plt.scatter(x1, y1, c='green', s=50, label='Keypoints 1')
        plt.scatter(x2, y2, c='red', s=50, label='Keypoints 2')

        plt.legend()

        plt.title('Keypoints on Image')
        plt.show()
