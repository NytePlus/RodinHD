import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    src_triplane_path = '/home/wcc/RodinHD/data/triplane_128_4/1_2.npy'
    with open(src_triplane_path, 'rb') as f:
        src_triplane = np.load(f)
        src_triplane = torch.as_tensor(src_triplane, dtype=torch.float32)
    data = src_triplane
    data = data.reshape(3, 32, 128, 128)
    data = torch.norm(data, dim = 1)
    data0, data1, data2 = data
    data0 = torch.flip(data0, dims = [0])
    data2 = torch.flip(data1, dims = [1])

    mask = data0.numpy() < 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 行 3 列的子图布局

    sns.heatmap(data0, annot=False, cmap='viridis', ax=axes[0])
    axes[0].set_title('dim 0')

    # 绘制第二张热力图
    sns.heatmap(data[1], annot=False, cmap='plasma', ax=axes[1])
    axes[1].set_title('dim 1')

    # 绘制第三张热力图
    sns.heatmap(data[2], annot=False, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('dim 2')

    plt.tight_layout()

    plt.savefig('test.png', dpi = 300, bbox_inches = 'tight')

    plt.close()