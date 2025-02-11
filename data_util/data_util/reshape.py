import os
import numpy as np
import torch
import torch.nn.functional as F

#! 请在项目根目录下运行

def downsample_npy_files(input_dir, output_dir, target_size=(128, 128)):
    """
    压缩文件夹下的所有 .npy 文件中的数据到指定大小，并保存到输出文件夹。

    Args:
        input_dir (str): 输入文件夹路径，包含原始 .npy 文件。
        output_dir (str): 输出文件夹路径，用于保存压缩后的 .npy 文件。
        target_size (tuple): 目标分辨率 (height, width)，默认 (128, 128)。
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入文件夹中的所有 .npy 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            # 加载 .npy 文件数据
            data = np.load(input_path)
            
            # 检查数据形状是否符合 [3, 1, 32, H, W]
            if len(data.shape) != 5 or data.shape[1:3] != (1, 32):
                print(f"Skipping file {file_name}: Unexpected shape {data.shape}")
                continue
            
            # 转换为 PyTorch Tensor
            tensor_data = torch.tensor(data)
            
            # 压缩每个平面到目标大小
            downsampled_tensor = F.interpolate(
                tensor_data.view(-1, 1, data.shape[-2], data.shape[-1]),
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
            # 恢复形状 [3, 1, 32, H, W]
            downsampled_tensor = downsampled_tensor.view(data.shape[0], 1, 32, target_size[0], target_size[1])
            
            # 转回 NumPy 数组
            downsampled_data = downsampled_tensor.numpy()
            
            # 保存压缩后的数据
            np.save(output_path, downsampled_data)
            print(f"Processed and saved: {output_path}")

def reshape():
    input_folder = "data/save_triplane_and_mlp4"  # 替换为实际的输入文件夹路径
    output_folder = "data/triplane_128_4"  # 替换为实际的输出文件夹路径
    downsample_npy_files(input_folder, output_folder, target_size=(128, 128))

def check_shape():
    import numpy as np

    # 替换为你的 .npy 文件路径
    file_path = "data/triplane_128/portrait3d_1.npy"

    # 加载 .npy 文件
    data = np.load(file_path)

    # 打印数据的 shape
    print("Shape of the .npy file:", data.shape)

def check_checkpoint():
    pass

if __name__ == '__main__':
    reshape()