import torch as th
from base_model import PreMultiscaleVAELatentUNet
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# pardir=os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# sys.path.append(os.path.join(pardir, 'Warper'))
# from Warper.modules.conv_warping import MultiscaleVAELatentUNet
# 创建一个形状为 (3, 1, 8, 128, 128) 的输入张量
input_tensor = th.randn(3, 1, 8, 128, 128)

# 创建一个形状为 (3, 5, 256, 256) 的 vae_ms_feature 张量列表
vae_ms_feature = [th.randn(3, 256, 256, 256) for _ in range(5)]



class_name = PreMultiscaleVAELatentUNet
model = class_name(
    in_channels=2,
    model_channels=32,
    out_channels=2,
    num_res_blocks=2,
    attention_resolutions=[32, 64],
    dropout=0.1,
    channel_mult=(1, 2, 4, 4, 8, 8, 16),
    use_fp16=False,
    num_heads=1,
    num_heads_upsample=-1,
    num_head_channels=8,
    use_scale_shift_norm=False,
    resblock_updown=False,
    encoder_channels=256,   
    )
# model = class_name(
#     in_channels=2,
#     model_channels=32,
#     out_channels=2,
#     num_res_blocks=2,
#     attention_resolutions=[32, 64],
#     dropout=0.1,
#     channel_mult=(1, 2, 4, 4, 8, 8, 16),
#     use_fp16=False,
#     num_heads=1,
#     num_heads_upsample=-1,
#     num_head_channels=8,
#     use_scale_shift_norm=False,
#     resblock_updown=False,
#     encoder_channels=256,
#     use_mask=True
#     )



# 将模型设置为评估模式
model.eval()

# 前向传播，获取输出
with th.no_grad():
    output, mask = model(input_tensor, vae_ms_feature)

# 打印输入和输出的形状
print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
if mask is not None:
    print("Mask shape:", mask.shape)
else:
    print("Mask is None")