import math
import torch as th

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torchck

from modules.nn import avg_pool_nd, conv_nd, normalization, timestep_embedding, zero_module, Conv3DAware, AttentionPooling, checkpoint
from modules.fp16_util import convert_module_to_f16, convert_module_to_f32

_FORCE_MEM_EFFICIENT_ATTN = 1
print('FORCE_MEM_EFFICIENT_ATTN=', _FORCE_MEM_EFFICIENT_ATTN, '@UNET:QKVATTENTION')
if _FORCE_MEM_EFFICIENT_ATTN:
    from xformers.ops import memory_efficient_attention  # noqa


class EmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, encoder_out=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            dropout=0,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            use_axis=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0),
            nn.Identity(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if use_axis:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
                nn.SiLU() if use_scale_shift_norm else nn.Identity(),
                nn.Dropout(p=dropout),
                # zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
                Conv3DAware(self.out_channels, self.out_channels)
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
                nn.SiLU() if use_scale_shift_norm else nn.Identity(),
                nn.Dropout(p=dropout),
                zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),

            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        return checkpoint(self._forward, [x], self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
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
            use_checkpoint=False,
            encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, swish=0.0)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1) # (bs, 256, -1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2

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


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder_channels=None,
            use_mask=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.encoder_channels = encoder_channels
        self.use_mask = use_mask

        if self.num_classes is not None:
            raise "Class embedding is not supported."

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            use_axis = True if level == 0 else False

            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_axis=use_axis,
                    )
                ]
                ch = int(mult * model_channels)
                print(ds, attention_resolutions)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_axis=False,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
            ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:

            use_axis = True if level == 0 else False
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_axis=use_axis,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_axis=False,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        if self.use_mask:
            self.mask = nn.Sequential(
                nn.Identity(),
                zero_module(conv_nd(dims, out_channels, 1, 3, padding=1)),
                nn.Sigmoid(),
            )

        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            raise "Class label is not supported."

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


# use this
class WarpingNetwork(nn.Module):
    def __init__(
            self,
            xf_width,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            use_fp16,
            num_heads,
            num_heads_upsample,
            num_head_channels,
            use_scale_shift_norm,
            resblock_updown,
            in_channels=32,
            use_mask=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        class_name = MultiscaleVAELatentUNet
        self.decoder = class_name(
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            encoder_channels=xf_width,
            use_mask=use_mask,
        )

    def forward(self, xt, vae_ms_feature=None):
        pred = self.decoder(xt, vae_ms_feature)
        return pred


class MultiscaleVAELatentUNet(UNetModel):
    def __init__(
            self,
            in_channels,
            *args,
            **kwargs,
    ):
        super().__init__(in_channels, *args, **kwargs)
        self.dtype = th.float32
        patch_size = 16
        encoder_dim = 64
        att_pool_heads = 8
        self.scaling_factor = 0.13025

        self.norm8 = normalization(256)
        self.norm16 = normalization(256)
        self.norm32 = normalization(256)
        self.norm64 = normalization(128)
        self.norm128 = normalization(64)
        self.vae_ms_feature_proj_8 = nn.Conv2d(256, self.encoder_channels, kernel_size=8, stride=8, bias=False)
        self.vae_ms_feature_proj_16 = nn.Conv2d(256, self.encoder_channels, kernel_size=4, stride=4, bias=False)
        self.vae_ms_feature_proj_32 = nn.Conv2d(256, self.encoder_channels, kernel_size=2, stride=2, bias=False)
        self.vae_ms_feature_proj_64 = nn.Conv2d(128, self.encoder_channels, kernel_size=1, stride=1, bias=False)
        self.vae_ms_feature_proj_128 = nn.Conv2d(64, self.encoder_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x, vae_ms_feature):
        '''
        latent_outputs: dict {'last_hidden_state': tensor, 'pooler_output': tensor}
        '''
        input_type = x.dtype
        x = x.type(self.dtype)
        hs = []

        vae_feature_8 = self.vae_ms_feature_proj_8(self.norm8(vae_ms_feature[4]).type(self.dtype))
        vae_feature_8 = vae_feature_8.reshape(vae_feature_8.shape[0], vae_feature_8.shape[1],
                                                -1)  # shape: (N, C, L)
        vae_feature_16 = self.vae_ms_feature_proj_16(self.norm16(vae_ms_feature[3]).type(self.dtype))
        vae_feature_16 = vae_feature_16.reshape(vae_feature_16.shape[0], vae_feature_16.shape[1],
                                                -1)  # shape: (N, C, L)
        vae_feature_32 = self.vae_ms_feature_proj_32(self.norm32(vae_ms_feature[2]).type(self.dtype))
        vae_feature_32 = vae_feature_32.reshape(vae_feature_32.shape[0], vae_feature_32.shape[1],
                                                -1)  # shape: (N, C, L)
        vae_feature_64 = self.vae_ms_feature_proj_64(self.norm64(vae_ms_feature[1]).type(self.dtype))
        vae_feature_64 = vae_feature_64.reshape(vae_feature_64.shape[0], vae_feature_64.shape[1],
                                                -1)  # shape: (N, C, L)
        vae_feature_128 = self.vae_ms_feature_proj_128(self.norm128(vae_ms_feature[0]).type(self.dtype))
        vae_feature_128 = vae_feature_128.reshape(vae_feature_128.shape[0], vae_feature_128.shape[1],
                                                  -1)  # shape: (N, C, L)

        # encoder_out_feature = {
        #     128: th.zeros(x.shape[0], self.encoder_channels, 128 * 128).half().to(x.device),
        #     64: th.zeros(x.shape[0], self.encoder_channels, 64 * 64).half().to(x.device),
        #     32: th.zeros(x.shape[0], self.encoder_channels, 32 * 32).half().to(x.device),
        #     16: th.zeros(x.shape[0], self.encoder_channels, 16 * 16).half().to(x.device),
        #     8: th.zeros(x.shape[0], self.encoder_channels, 8 * 8).half().to(x.device)}
        encoder_out_feature = {
            # 128: vae_feature_128,
            64: vae_feature_64,
            32: vae_feature_32,
            16: vae_feature_16,
            8: vae_feature_8,
        }

        h = x.type(self.dtype)
        for module in self.input_blocks:
            encoder_out = encoder_out_feature[h.shape[-2]] if h.shape[-2] in encoder_out_feature else None
            h = module(h, encoder_out)
            hs.append(h)

        encoder_out = encoder_out_feature[h.shape[-2]] if h.shape[-2] in encoder_out_feature else None
        h = self.middle_block(h, encoder_out)

        for module in self.output_blocks:
            encoder_out = encoder_out_feature[h.shape[-2]] if h.shape[-2] in encoder_out_feature else None
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, encoder_out)
        h = h.type(input_type)
        h = self.out(h)

        mask = self.mask(h) if self.use_mask else None
        return h.type(input_type), mask


