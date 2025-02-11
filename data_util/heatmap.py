import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn import MSELoss

if __name__ == '__main__':
    src_triplane_path = '/home/wcc/RodinHD/data/save_triplane_and_mlp5/portrait3d_1.npy'
    tgt_triplane_path = '/home/wcc/RodinHD/data/save_triplane5/portrait3d_1_思考.npy'
    warp_path = '/data1/wcc/RodinHD/data/save_warping_module/portrait3d_1_思考.pt'
    warp2_path = '/data1/wcc/RodinHD/data/save_warping_module2/validation/portrait3d_1_思考.npy'
    with open(src_triplane_path, 'rb') as f:
        src_triplane = np.load(f)
        src_triplane = torch.as_tensor(src_triplane, dtype=torch.float32)

    with open(tgt_triplane_path, 'rb') as f:
        tgt_triplane = np.load(f)
        tgt_triplane = torch.as_tensor(tgt_triplane, dtype=torch.float32)

    warp_triplane = torch.load(warp_path, map_location='cpu').detach()
    warp_triplane = torch.stack([warp_triplane[..., :512], warp_triplane[..., 512:1024], warp_triplane[..., 1024:]]).unsqueeze(1)

    with open(warp2_path, 'rb') as f:
        warp2_triplane = np.load(f)
        warp2_triplane = torch.as_tensor(warp2_triplane, dtype=torch.float32)

    loss = MSELoss()
    print(loss(tgt_triplane, warp_triplane), loss(tgt_triplane, warp2_triplane))

    data = tgt_triplane
    data = data.reshape(3, 32, 512, 512)
    data = torch.norm(data, dim = 1)
    data0, data1, data2 = data
    data0 = torch.flip(data0, dims = [0])
    data2 = torch.flip(data2, dims = [1])


    mask0 = data0.numpy() < -1
    mask1 = data1.numpy() < -1
    mask2 = data2.numpy() < -1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 行 3 列的子图布局

    sns.heatmap(data0, mask=mask0, annot=False, cmap='viridis', ax=axes[0])
    axes[0].set_title('dim 0')

    # 绘制第二张热力图
    sns.heatmap(data[1], mask=mask1, annot=False, cmap='plasma', ax=axes[1])
    axes[1].set_title('dim 1')

    # 绘制第三张热力图
    sns.heatmap(data[2], mask=mask2, annot=False, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('dim 2')

    plt.tight_layout()

    plt.savefig('src5.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    plt.close()