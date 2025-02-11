import torch

# 加载 .ckpt 文件
ckpt_path = "data/save_triplane_and_mlp2/portrait3d_1_iwc_state.ckpt"
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

# 打印 checkpoint 的 keys
print("Keys in the checkpoint:")
print(checkpoint.keys())

# 如果有 'state_dict'，可以打印它的 keys（通常是模型参数）
if 'state_dict' in checkpoint:
    print("\nKeys in state_dict:")
    print(checkpoint['state_dict'].keys())
