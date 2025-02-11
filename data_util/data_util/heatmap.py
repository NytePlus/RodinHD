import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch.nn import MSELoss

def clip(data):
    data0, data1, data2 = data
    data0 = data0[0, :, 256: 448, 160: 352]
    data1 = data1[0, :, 192: 384, 160: 352]
    data2 = data2[0, :, 192: 384, 256: 448]

    edit = torch.stack([data0, data1, data2])
    edit = torch.nn.functional.interpolate(edit, size=(512, 512), mode='bilinear')
    edit = edit.reshape(3, 1, 32, 512, 512)
    return edit

def clip_and_save(data, save_dir):
    data0, data1, data2 = data
    data0 = data0[..., 256: 448, 160: 352]
    data1 = data1[..., 192: 384, 160: 352]
    data2 = data2[..., 192: 384, 256: 448]

    edit = torch.stack([data0, data1, data2])
    edit = torch.nn.functional.interpolate(edit, size=(512, 512), mode='bilinear')
    edit = edit.reshape(3, 1, 32, 512, 512)
    np.save(save_dir, edit.detach().cpu().numpy())


if __name__ == '__main__':
    src_triplane_path = '/home/wcc/RodinHD/data/save_triplane_and_mlp4/1.npy'
    tgt_triplane_path = '/home/wcc/RodinHD/data/save_triplane_and_mlp4/1_2.npy'
    warp_path = '/home/wcc/RodinHD/data/save_warping_module/validation/1_2.npy'
    warp2_path = '/home/wcc/RodinHD/data/save_warping_module/validation/1_2.npy'
    with open(src_triplane_path, 'rb') as f:
        src_triplane = np.load(f)
        src_triplane = torch.as_tensor(src_triplane, dtype=torch.float32)

    with open(tgt_triplane_path, 'rb') as f:
        tgt_triplane = np.load(f)
        tgt_triplane = torch.as_tensor(tgt_triplane, dtype=torch.float32)

    if warp_path:
        warp_triplane = np.load(warp_path)
        warp_triplane = torch.as_tensor(warp_triplane, dtype=torch.float32).reshape(tgt_triplane.shape)

    with open(warp2_path, 'rb') as f:
        warp2_triplane = np.load(f)
        warp2_triplane = torch.as_tensor(warp2_triplane, dtype=torch.float32).reshape(32, 512, 3 * 512)
        split = torch.split(warp2_triplane, 512, dim = -1)
        warp2_triplane = torch.stack(split, dim = 0)
        warp2_triplane = warp2_triplane.unsqueeze(1)
        # FIXME:
        # np.save('/data/save_warping_module/1_2.npy', warp2_triplane.cpu().detach().numpy())

    loss = MSELoss()
    tgt_triplane = clip(tgt_triplane)
    print(loss(tgt_triplane, warp_triplane))
    data = (tgt_triplane - warp_triplane).reshape(3, 32, 512, 512)

    data = torch.norm(data, dim = 1)
    data0, data1, data2 = data
    data0 = torch.flip(data0, dims = [0])
    data2 = torch.flip(data2, dims = [1])

    mask0 = data0.numpy() < 0.4
    mask1 = data1.numpy() < 0.4
    mask2 = data2.numpy() < 0.4

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 行 3 列的子图布局

    sns.heatmap(data0, mask=mask0, annot=False, cmap='viridis', ax=axes[0])
    axes[0].set_title('dim 0')

    # 绘制第二张热力图
    sns.heatmap(data1, mask=mask1, annot=False, cmap='plasma', ax=axes[1])
    axes[1].set_title('dim 1')

    # 绘制第三张热力图
    sns.heatmap(data2, mask=mask2, annot=False, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('dim 2')

    plt.tight_layout()

    plt.savefig('src[...,0].png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.close()
