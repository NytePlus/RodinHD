import torch
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    occ_path = '/home/wcc/RodinHD/data_util/occ.pt'
    occ = torch.load(occ_path, map_location='cpu').detach()
    occ = occ.reshape(3, 512, 512).flip(dims = [1])

    plt.figure(figsize=(10, 8))
    occ_data = occ[0].numpy()  # 转换为 NumPy 数组

    # 绘制热力图
    sns.heatmap(occ_data, cmap='viridis', cbar=True, square=True)

    # 添加标题和标签
    plt.title("Occ Heatmap Visualization")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")

    # 显示图像
    plt.show()