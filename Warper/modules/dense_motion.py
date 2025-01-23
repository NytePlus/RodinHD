# coding: utf-8

"""
The module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
"""

from torch import nn
import torch.nn.functional as F
import torch
from Warper.modules.util import Hourglass, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress, down_scale, estimate_occlusion_map=True):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*compress, max_features=max_features, num_blocks=num_blocks)  # ~60+G
        self.down_scale = down_scale

        # self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)  # 65G! NOTE: computation cost is large
        self.compress = nn.Conv2d(feature_channel, compress, kernel_size=down_scale, stride=down_scale)  # 0.8G
        self.norm = nn.BatchNorm2d(compress, affine=True)
        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, _, c, h, w = feature.shape # (bs, 3, 1, 32, 512, 512)
        xy_grid, yz_grid, xz_grid = make_coordinate_grid((h, h, w), ref=kp_source)  # 3 * (512, 512, 2)

        xy_grid = xy_grid.view(1, 1, h, w, 2)
        yz_grid = yz_grid.view(1, 1, h, h, 2)
        xz_grid = xz_grid.view(1, 1, h, w, 2)  # (1, 1, h=512, w=512, 2)

        kp_driving_xy = kp_driving[..., : 2].view(bs, self.num_kp, 1, 1, 2)
        kp_driving_yz = kp_driving[..., 1 :].view(bs, self.num_kp, 1, 1, 2)
        kp_driving_xz = kp_driving[..., ::2].view(bs, self.num_kp, 1, 1, 2)

        kp_source_xy = kp_source[..., : 2].view(bs, self.num_kp, 1, 1, 2)
        kp_source_yz = kp_source[..., 1 :].view(bs, self.num_kp, 1, 1, 2)
        kp_source_xz = kp_source[..., ::2].view(bs, self.num_kp, 1, 1, 2)

        # NOTE: there lacks an one-order flow
        driving_to_source_xy = xy_grid - kp_driving_xy + kp_source_xy
        driving_to_source_yz = yz_grid - kp_driving_yz + kp_source_yz
        driving_to_source_xz = xz_grid - kp_driving_xz + kp_source_xz  # (bs, num_kp, h, w, 2)

        # adding background feature
        xy_grid = xy_grid.repeat(bs, 1, 1, 1, 1)
        yz_grid = yz_grid.repeat(bs, 1, 1, 1, 1)
        xz_grid = xz_grid.repeat(bs, 1, 1, 1, 1)

        sparse_motions_xy = torch.cat([xy_grid, driving_to_source_xy], dim=1)
        sparse_motions_yz = torch.cat([yz_grid, driving_to_source_yz], dim=1)
        sparse_motions_xz = torch.cat([xz_grid, driving_to_source_xz], dim=1)  # (bs, 1+num_kp, h, w, 2)

        del xy_grid, yz_grid, xz_grid, driving_to_source_xy, driving_to_source_yz, driving_to_source_xz

        sparse_motions = torch.stack([sparse_motions_xy, sparse_motions_yz, sparse_motions_xz]).permute(1, 0, 2, 3, 4, 5)
        del sparse_motions_xy, sparse_motions_yz, sparse_motions_xz

        return sparse_motions.contiguous()

    def create_deformed_feature(self, feature, sparse_motions):
        # 4028MB
        bs, _, _, c, h, w = feature.shape # (bs, 3, 1, 32, 512, 512)
        feature_repeat = feature.permute(1, 0, 2, 3, 4, 5).repeat(1, 1, self.num_kp+1, 1, 1, 1)      # (3, bs, num_kp+1, c, h, w)
        feature_repeat = feature_repeat.view(3, bs * (self.num_kp+1), c, h, w)                      # (3, bs*(num_kp+1), c, h, w)
        f_xy, f_yz, f_xz = feature_repeat

        m_xy = sparse_motions[:, 0].reshape((bs * (self.num_kp+1), h, w, 2))   # (bs*(num_kp+1), h, w, 2)
        m_yz = sparse_motions[:, 1].reshape((bs * (self.num_kp+1), h, h, 2))
        m_xz = sparse_motions[:, 2].reshape((bs * (self.num_kp+1), h, w, 2))

        d_xy = F.grid_sample(f_xy, m_xy, align_corners=False) # (bs*(num_kp+1), c, h, w)
        d_yz = F.grid_sample(f_yz, m_yz, align_corners=False)
        d_xz = F.grid_sample(f_xz, m_xz, align_corners=False)

        # NOTE: 20922MB
        del sparse_motions, feature_repeat, m_xy, m_yz, m_xz, f_xy, f_yz, f_xz

        # NOTE: 11600MB
        sparse_deformed = torch.stack([d_xy, d_yz, d_xz]).view((3, bs, self.num_kp+1, c, h, w))           # (3, bs, num_kp+1, c, h, w)
        del d_xy, d_yz, d_xz

        # NOTE: 11600MB
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        c, h, w = feature.shape[3:]  # (c=32, h=512, w=512)
        spatial_size = [h, h, w]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        heatmap = gaussian_driving - gaussian_source  # (bs, num_kp, d, h, w)

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.dtype).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, 1+num_kp, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, _, c, h, w = feature.shape # (bs, 3, 1, 32, 512, 512)

        out_dict = dict()

        # 1. deform 3d feature
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, 3, 1+num_kp, h, w, 2)
        # NOTE: 12474MB
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (3, bs, num_kp+1, c = 32, h = 512, w = 512)

        # TODO: Compress is necessary because mm limit.
        deformed_feature = deformed_feature.view(-1, c, h, w)
        deformed_feature = self.compress(deformed_feature)  # (bs*3*(num_kp+1), 16, 128, 128)
        deformed_feature = self.norm(deformed_feature)  # (bs*3*(num_kp+1), 16, 128, 128)
        deformed_feature = F.relu(deformed_feature)  # (bs*3*(num_kp+1), 16, 128, 128)

        # TODO: Heat map is a 3D concept, how to adjust it for triplane?
        # 2. (bs, 1+num_kp, d, h, w)
        # heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)  # (bs, 1+num_kp, 1, d, h, w)

        # input = torch.cat([heatmap, deformed_feature], dim=2)  # (bs, 1+num_kp, c=5, d=512, h=512, w=512)
        _, _, h_c, w_c = deformed_feature.shape
        input = deformed_feature
        input = input.view(bs*3, -1, h_c, w_c)  # (bs*3, (1+num_kp)*c=22*4, h=128, w=128)

        prediction = self.hourglass(input)  # (bs*3, (1+num_kp)*c+8, h, w)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)  # (bs * 3, 1+num_kp, h=128, w=128)
        mask = F.interpolate(mask, scale_factor=(self.down_scale, self.down_scale))
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs * 3, num_kp+1, 1, h, w)
        sparse_motion = sparse_motion.view(bs * 3, -1, h, w, 2).permute(0, 1, 4, 2, 3)    # (bs * 3, num_kp+1, 2, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs * 3, 2, h, w)  mask take effect in this place
        deformation = deformation.permute(0, 2, 3, 1)           # (bs * 3, h, w, 2)

        out_dict['deformation'] = deformation

        # NOTE: disabled
        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape
            prediction_reshape = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # Bx1x64x64
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

    def forward2(self, feature, kp_driving, kp_source):
        bs, _, _, c, h, w = feature.shape # (bs, 3, 1, 32, 512, 512)

        out_dict = dict()

        # 1. deform 3d feature
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, 3, 1+num_kp, h, w, 2)
        # NOTE: 12474MB
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (3, bs, num_kp+1, c = 32, h = 512, w = 512)

        deformed_feature = deformed_feature.view(-1, c, h, w)
        deformed_feature = self.compress(deformed_feature)  # (bs*3*(num_kp+1), 16, 128, 128)
        deformed_feature = self.norm(deformed_feature)  # (bs*3*(num_kp+1), 16, 128, 128)
        deformed_feature = F.relu(deformed_feature)  # (bs*3*(num_kp+1), 16, 128, 128)

        feature = feature.view(-1, c, h, w)
        feature = self.compress(feature)  # (bs*3, 16, 128, 128)
        feature = self.norm(feature)  # (bs*3, 16, 128, 128)
        feature = F.relu(feature)  # (bs*3, 16, 128, 128)

        _, c_c, h_c, w_c = deformed_feature.shape
        # TODO: Compress is necessary because mm limit.
        key = deformed_feature.permute(0, 2, 3, 1).contiguous().view(-1, c)
        query = feature.view(-1, c_c)
        key = key.view(bs*3, -1, h_c, w_c, c_c).permute(0, 1, 4, 2, 3)
        query = query.view(-1, h_c, w_c, c_c).permute(0, 3, 1, 2).unsqueeze(1)
        value = (key * query).sum(dim = 2)

        mask = value.view(bs*3, -1, h_c, w_c)
        mask = F.softmax(mask, dim=1)  # (bs * 3, 1+num_kp, h=128, w=128)
        mask = F.interpolate(mask, scale_factor=(self.down_scale, self.down_scale))
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs * 3, num_kp+1, 1, h, w)
        sparse_motion = sparse_motion.view(bs * 3, -1, h, w, 2).permute(0, 1, 4, 2, 3)    # (bs * 3, num_kp+1, 2, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs * 3, 2, h, w)  mask take effect in this place
        deformation = deformation.permute(0, 2, 3, 1)           # (bs * 3, h, w, 2)

        out_dict['deformation'] = deformation

        # NOTE: disabled
        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape
            prediction_reshape = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # Bx1x64x64
            out_dict['occlusion_map'] = occlusion_map

        return out_dict