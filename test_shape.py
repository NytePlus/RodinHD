import pretrained_diffusion.base_model
from pretrained_diffusion.base_model import BaseDiffusion
from pretrained_diffusion.base_model import BaseMultiscaleVAELatentUNet

import torch as th

# 超参数
in_channels = 32  # 输入通道数
model_channels = 128  # 模型的通道数
out_channels = 3  # 输出通道数 (例如 RGB 图像)
num_res_blocks = 2  # 残差块数量
attention_resolutions = [32, 16, 8]  # 注意力分辨率
dropout = 0.1  # dropout 概率
channel_mult = [1, 2, 4, 8]  # 每层的通道数倍增因子
use_fp16 = True  # 是否使用半精度
num_heads = 8  # 注意力头数
num_heads_upsample = 4  # 上采样的注意力头数
num_head_channels = 64  # 每个头的通道数
use_scale_shift_norm = True  # 是否使用尺度偏移规范化
resblock_updown = True  # 是否在上采样/下采样时使用残差块
encoder_channels = 64  # 假设编码器通道数为 64

# 实例化 BaseMultiscaleVAELatentUNet
model = BaseMultiscaleVAELatentUNet(
    in_channels=in_channels,
    model_channels=model_channels,
    out_channels=out_channels,
    num_res_blocks=num_res_blocks,
    attention_resolutions=attention_resolutions,
    dropout=dropout,
    channel_mult=channel_mult,
    use_fp16=use_fp16,
    num_heads=num_heads,
    num_heads_upsample=num_heads_upsample,
    num_head_channels=num_head_channels,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
    encoder_channels=encoder_channels
)


# 创建模型实例
model = BaseMultiscaleVAELatentUNet(
    in_channels=32,
    model_channels=128,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=[32, 16, 8],
    dropout=0.1,
    channel_mult=[1, 2, 4, 8],
    use_fp16=True,
    num_heads=8,
    num_heads_upsample=4,
    num_head_channels=64,
    use_scale_shift_norm=True,
    resblock_updown=True,
    encoder_channels=64
)

# 准备输入数据
xt = th.randn(1, 32, 128, 128)  # 输入的图像张量
timesteps = th.randint(0, 1000, (1,))  # 随机生成时间步张量
latent_outputs = {
    'last_hidden_state': th.randn(1, 64, 8, 8),
    'pooler_output': th.randn(1, 64)
}
vae_ms_feature = [
    th.randn(1, 128, 16, 16),  # 第一尺度特征
    th.randn(1, 256, 8, 8),    # 第二尺度特征
    th.randn(1, 512, 4, 4)     # 第三尺度特征
]

# 使用模型进行前向传播
output = model(xt, timesteps, latent_outputs, vae_ms_feature)

# 输出结果
print(output.shape)  # 查看输出的形状
