# coding: utf-8

"""
Warping field estimator(W) defined in the paper, which generates a warping field using the implicit
keypoint representations x_s and x_d, and employs this flow field to warp the source feature volume f_s.
"""

from torch import nn
import torch.nn.functional as F
from Warper.modules.util import SameBlock2d
from Warper.modules.dense_motion import DenseMotionNetwork


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp,
        block_expansion,
        max_features,
        num_down_blocks,
        reshape_channel,
        estimate_occlusion_map=False,
        dense_motion_params=None,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
        else:
            self.dense_motion_network = None

        self.third = SameBlock2d(block_expansion * (2 ** num_down_blocks), block_expansion * (2 ** num_down_blocks), kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks), out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map

    def deform_input(self, inp, deformation):
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, feature_3d, kp_driving, kp_source):
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            dense_motion = self.dense_motion_network.forward(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source  # Bx32x64x64
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']  # (Bx3)x512x512x2
            bs, _, _, c, h, w = feature_3d.shape    # Bx3x1x32x512x512
            feature_3d = feature_3d.view(bs * 3, c, h, w)
            out = self.deform_input(feature_3d, deformation)  # (Bx3)x32x512x512

            out = out.view(bs*3, c, h, w)  # -> (Bx3)x32x512x512
            out = self.third(out)  # -> (Bx3)x32x512x512
            out = self.fourth(out)  # -> (Bx3)x32x512x512

            if self.flag_use_occlusion_map and (occlusion_map is not None):
                out = out * occlusion_map

        ret_dct = {
            'occlusion_map': occlusion_map,
            'deformation': deformation,
            'out': out,
        }

        return ret_dct

    def _forward(self, feature_3d, kp_driving, kp_source):
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            dense_motion = self.dense_motion_network.forward(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source  # Bx32x64x64
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            out = dense_motion['deformation']  # (Bx3)x32x512x512

            bs, _, _, c, h, w = feature_3d.shape    # Bx3x1x32x512x512
            out = out.reshape(bs*3, c, h, w)  # -> (Bx3)x32x512x512
            out = self.third(out)  # -> (Bx3)x32x512x512
            out = self.fourth(out)  # -> (Bx3)x32x512x512

            if self.flag_use_occlusion_map and (occlusion_map is not None):
                out = out * occlusion_map

        ret_dct = {
            'occlusion_map': occlusion_map,
            'out': out,
        }

        return ret_dct