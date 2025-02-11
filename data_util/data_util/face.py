import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 加载 Haar 特征分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图像
    image = cv2.imread('/home/wcc/RodinHD/data/portrait3dexp_data/portrait3d_1_大笑/img_proc_fg_000000.png')

    # 检查图像是否正确加载
    if image is None:
        raise ValueError("图像未正确加载，请检查路径")

    # 检查图像数据类型并转换为 uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    scaleFactor = 1.1  # 必须大于 1
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5, minSize=(30, 30))

    # 绘制检测到的人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 将 BGR 图像转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 matplotlib 显示图像
    plt.imshow(image_rgb)
    plt.axis('off')  # 关闭坐标轴
    plt.title('Detected Faces')
    plt.show()