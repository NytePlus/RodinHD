import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import math
from .nn import linear, timestep_embedding, conv_nd, Conv3DAware, get_embedder, make_grid, AttentionPooling, zero_module, checkpoint
from .fp16_util import convert_module_to_f16, convert_module_to_f32

_FORCE_MEM_EFFICIENT_ATTN = 0
print('FORCE_MEM_EFFICIENT_ATTN=', _FORCE_MEM_EFFICIENT_ATTN, '@UNET:QKVATTENTION')
if _FORCE_MEM_EFFICIENT_ATTN:
    from xformers.ops import memory_efficient_attention  # noqa

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, dtype=None):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, dtype=dtype)

    def forward(self, x):
        y = super().forward(x).to(x.dtype)
        return y

class CrossAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            disable_self_attention=False,
            encoder_channels=None,
            dtype=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.disable_self_attention = disable_self_attention
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = channels // num_head_channels
        self.norm = GroupNorm32(num_channels=channels, num_groups=32, dtype=dtype)
        self.qkv = conv_nd(1, channels, channels * 3, 1, dtype=self.dtype)
        if self.disable_self_attention:
            self.qkv = conv_nd(1, channels, channels, 1, dtype=self.dtype)
        else:
            self.qkv = conv_nd(1, channels, channels * 3, 1, dtype=self.dtype)
        self.attention = QKVCrossAttention(self.num_heads, disable_self_attention=disable_self_attention)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1, dtype=self.dtype)
            self.norm_encoder = GroupNorm32(num_channels=channels, num_groups=32, dtype=dtype)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1, dtype=self.dtype))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            # from imagen article: https://arxiv.org/pdf/2205.11487.abs
            encoder_out = self.norm_encoder(encoder_out)
            # # #
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVCrossAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, disable_self_attention=False):
        super().__init__()
        self.n_heads = n_heads
        self.disable_self_attention = disable_self_attention

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        if self.disable_self_attention:
            ch = width // (1 * self.n_heads)
            q, = qkv.reshape(bs * self.n_heads, ch * 1, length).split(ch, dim=1)
        else:
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            if self.disable_self_attention:
                k, v = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            else:
                ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
                k = th.cat([ek, k], dim=-1)
                v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        if _FORCE_MEM_EFFICIENT_ATTN:
            q, k, v = map(lambda t: t.permute(0, 2, 1).contiguous(), (q, k, v))
            a = memory_efficient_attention(q, k, v)
            a = a.permute(0, 2, 1)
        else:
            weight = th.einsum(
                'bct,bcs->bts', q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)


class EmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, encoder_out=None):
        for layer in self:
            if isinstance(layer, CrossAttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv_nd(2, n_feats, 4 * n_feats, 3, padding=1))

                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def nonlinearity(x):
    # swish
    return x * th.sigmoid(x)


def swish_0(x):
    return x * F.sigmoid(x * float(0))


def Normalize(in_channels, num_groups=32):
    return th.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, ):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = th.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=0)

    def forward(self, x):
        input_type = x.dtype
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = th.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = th.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        if self.with_conv:
            self.conv = th.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

    def forward(self, x):
        input_type = x.dtype
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, use_3d_conv=False, use_checkpoint=False, use_scale_shift_norm=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = Normalize(in_channels)
        self.conv1 = th.nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        if temb_channels > 0:
            self.temb_proj = th.nn.Linear(temb_channels,
                                          2 * out_channels if use_scale_shift_norm else out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = th.nn.Dropout(dropout)
        # TODO: add conv axis
        if use_3d_conv:
            self.conv2 = Conv3DAware(out_channels, out_channels)
        else:
            self.conv2 = th.nn.Conv2d(out_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = th.nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1)
            else:
                self.nin_shortcut = th.nn.Conv2d(in_channels,
                                                 out_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0)

    def forward(self, x, temb=None):
        if temb is not None:
            return checkpoint(self._forward, (x, temb), self.parameters(), self.use_checkpoint)
        else:
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            if not self.use_scale_shift_norm:
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
            else:
                emb_out = self.temb_proj(nonlinearity(temb))[:, :, None, None]

        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.norm2(h)
            h = swish_0(h)
            h = h * (scale + 1) + shift
            h = self.dropout(h)
            h = self.conv2(h)
        else:
            h = self.norm2(h)
            h = nonlinearity(h)
            h = self.dropout(h)
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class LightWarpingNetwork(nn.Module):
    def __init__(self, scale=4, image_size=256, n_feats=64, n_resblocks=6, kernel_size=3, use_fp16=False,
                 use_checkpoint=False, ch_mult=[1, 2], use_scale_shift_norm=False, use_3d_conv=True,
                 condition_channels=32, dtype="32"):
        super(LightWarpingNetwork, self).__init__()

        self.n_feats = n_feats
        self.temb_ch = n_feats * 4
        self.num_res_blocks = 1
        self.num_resolutions = len(ch_mult)
        self.dtype = th.float16 if dtype == "16" else th.float32
        self.use_checkpoint = use_checkpoint
        self.attention_level = (2,)
        print("use_checkpoint", use_checkpoint)
        ch = n_feats

        in_ch_mult = [1, ] + list(ch_mult)
        resamp_with_conv = True
        use_3d_conv = use_3d_conv

        self.input_layer = EmbedSequential(conv_nd(2, 32, n_feats, kernel_size, padding=(kernel_size // 2)),
                                                   ResnetBlock(in_channels=n_feats, out_channels=ch * in_ch_mult[0],
                                                               temb_channels=self.temb_ch, dropout=0,
                                                               use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint,
                                                               use_scale_shift_norm=use_scale_shift_norm)
                                                   )

        # Build encoder
        self.down = nn.ModuleList()
        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if i_level == 0 and i_block == 0:
                    block_in += condition_channels
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=0,
                                         use_3d_conv=use_3d_conv,
                                         use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out

            down = nn.Module()
            down.block = EmbedSequential(*block)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # define body module
        m_body = []
        for _ in range(n_resblocks // 2):
            m_body.extend([
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=0,
                            use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm),
                CrossAttentionBlock(block_in, num_head_channels=32, disable_self_attention=True,
                                    encoder_channels=self.temb_ch, dtype=self.dtype),
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=0,
                            use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm),
            ])

        # Build decoder
        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                if i_level == 0 and i_block == self.num_res_blocks:
                    skip_in += condition_channels
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=0,
                                         use_3d_conv=use_3d_conv,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out

            up = nn.Module()
            up.block = EmbedSequential(*block)
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # define tail module
        m_tail = [
            ResnetBlock(in_channels=block_in + n_feats + condition_channels, out_channels=n_feats,
                        temb_channels=self.temb_ch, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm),
            ResnetBlock(in_channels=n_feats, out_channels=32, temb_channels=self.temb_ch, dropout=0,
                        use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm),
        ]

        patch_size = 16
        encoder_dim = 64
        att_pool_heads = 8
        self.scaling_factor = 0.13025

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=encoder_dim, kernel_size=patch_size, stride=patch_size,
                               bias=False)
        if encoder_dim != self.temb_ch:
            self.encoder_proj = nn.Linear(encoder_dim, self.temb_ch)
        else:
            self.encoder_proj = nn.Identity()

        self.encoder_pooling = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            AttentionPooling(att_pool_heads, encoder_dim),
            nn.Linear(encoder_dim, n_feats * 4),
            nn.LayerNorm(n_feats * 4)
        )
        self.transformer_proj = nn.Identity()
        self.body = EmbedSequential(*m_body)
        self.tail = EmbedSequential(*m_tail)
        self.out = ResnetBlock(in_channels=condition_channels, out_channels=32, temb_channels=0, dropout=0,
                               use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint,
                               use_scale_shift_norm=use_scale_shift_norm)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_layer.apply(convert_module_to_f16)
        self.down.apply(convert_module_to_f16)
        self.body.apply(convert_module_to_f16)
        self.up.apply(convert_module_to_f16)
        self.tail.apply(convert_module_to_f16)
        if not self.rezero:
            self.out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_layer.apply(convert_module_to_f32)
        self.down.apply(convert_module_to_f32)
        self.body.apply(convert_module_to_f32)
        self.up.apply(convert_module_to_f32)
        self.tail.apply(convert_module_to_f32)
        if not self.rezero:
            self.out.apply(convert_module_to_f32)

    def forward(self, xt, ref):
        # lightweight encoder
        input_type = xt.dtype
        xt = xt.type(self.dtype)

        # 966 MB
        latent_outputs = (ref * self.scaling_factor).type(self.dtype)
        latent_outputs_emb = self.conv1(latent_outputs)  # shape = [*, width, grid, grid]
        latent_outputs_emb = latent_outputs_emb.reshape(latent_outputs_emb.shape[0], latent_outputs_emb.shape[1],
                                                        -1)  # shape = [*, width, grid ** 2]
        latent_outputs_emb = latent_outputs_emb.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        encoder_out = self.encoder_proj(latent_outputs_emb)
        encoder_out = encoder_out.permute(0, 2, 1)  # NLC -> NCL
        encoder_out = encoder_out.type(self.dtype)

        # 3018 MB
        x = self.input_layer(xt)
        x = th.cat([x, xt], dim=1)

        # 3402 MB
        hs = [x]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 10672 MB
        h = hs[-1]
        h = self.body(h, encoder_out)

        # 10672 MB
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](th.cat([h, hs.pop()], dim=1))
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 26634 MB -> 19024 MB
        # add low_res
        x = self.tail(th.cat([h, x], dim=1))
        output = self.out(x)

        return output.type(input_type)
