import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 假设有一个 (3, 22, 1, 512, 512) 的 tensor
    tensor = torch.load('mask4.pt')  # 替换为你的实际数据

    slice_0 = tensor[0]  # (22, 1, 512, 512)
    slice_0 = torch.flip(slice_0, dims = [1])

    # 在 dim1 上找到最大值的索引，形状为 (1, 512, 512)
    max_indices = torch.argmax(slice_0, dim=0).squeeze()  # (512, 512)

    # 定义颜色映射
    def index_to_color(index):
        # 定义颜色映射规则
        color = [0, 0, 0]
        bc = 255 if index % 2 == 0 else 127
        index /= 2
        if index % 2 == 1:
            color[0] = bc
        if index / 2 % 2 == 1:
            color[1] = bc
        if index / 4 % 2 == 1:
            color[2] = bc
        return color

    # 将索引映射到颜色
    height, width = max_indices.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)  # 创建 RGB 图像

    for i in range(height):
        for j in range(width):
            color_image[i, j] = index_to_color(max_indices[i, j].item())

    # 可视化
    plt.imshow(color_image)
    plt.title("Max Indices Visualization")
    plt.show()