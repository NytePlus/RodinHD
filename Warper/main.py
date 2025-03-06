import os
import sys

import tyro
from LivePortrait.live_portrait_pipeline import LivePortraitPipeline
from LivePortrait.config.crop_config import CropConfig
from LivePortrait.config.inference_config import InferenceConfig
from LivePortrait.config.argument_config import ArgumentConfig

sys.path.append('../Renderer')
import torch
import argparse
import numpy as np

from torch import optim, nn
from mpi4py import MPI
from TriplaneFit.network import NeRFNetwork

import dist_util

from provider import TriplaneLatentDataset, TriplaneDataset, TriplaneImagesDataset, TriplaneFeatureDataset
from metrics import PSNRMeter, CustomMSELoss
from modules.motion_extractor import MotionExtractor
from trainer import seed_everything, Trainer
from modules.warping_network import WarpingNetwork as _WarpingNetwork
from modules.conv_warping import WarpingNetwork
from modules.conv_light_warping import LightWarpingNetwork
from modules.discriminator import Discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--src_root', type=str, default='')
    parser.add_argument('--src_data', type=str, default='')
    parser.add_argument('--tgt_root', type=str, default='')
    parser.add_argument('--tgt_data', type=str, default='')
    parser.add_argument('--latent_root', type=str, default='')
    parser.add_argument('--ms_feature_root', type=str, default='')
    parser.add_argument('--latent_type', type=str, default='emo')
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('-debug', action='store_true', help="debug for tensor dims")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='facescape')
    ### training options
    parser.add_argument('--num_epochs', type=int, default=9000, help="training iters")
    parser.add_argument('--lr0', type=float, default=2e-4, help="initial learning rate for warper")
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--r_ckpt', type=str, default='/home/wcc/RodinHD/data/save_triplane_and_mlp2/checkpoints/ngp_ep0017.pth.pth')
    parser.add_argument('--num_rays', type=int, default=8192, help="num rays sampled per image for each training step")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")


    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--cp', action='store_true', help="use TensorCP")
    parser.add_argument('--triplane_channels', type=int, default=32)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--resolution0', type=int, default=512)
    parser.add_argument('--resolution1', type=int, default=512)
    parser.add_argument('--grid_size', type=int, default=256)

    ### dataset options
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")

    ### experimental
    parser.add_argument('--eval_freq', type=int, default=10, help="eval freq")
    parser.add_argument('--no_tqdm', action='store_true', help="disable tqdm")
    parser.add_argument('--start_idx', type=int, default=-1, help="start idx of index file, <0 uses no start idx")
    parser.add_argument('--end_idx', type=int, default=-1, help="end idx of index file, <0 uses no end idx")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")

    opt = parser.parse_args()
    seed_everything(opt.seed)
    dist_util.setup_dist()
    device = dist_util.dev()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    renderer = NeRFNetwork(
        resolution=[opt.resolution0] * 3,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        grid_size=opt.grid_size,
        sigma_rank=[int(opt.triplane_channels // 4)] * 3,
        color_rank=[int(opt.triplane_channels // 4 * 3)] * 3,
        triplane_channels=opt.triplane_channels,
    )

    discriminator, optimizer_d, motion_extractor, optimizer_me = None, None, None, None
    if False:
        def partial_fields(target_class, kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('__') and hasattr(target_class, k)}
            return target_class(**filtered_kwargs)

        inference_cfg = partial_fields(InferenceConfig, ArgumentConfig.__dict__)
        crop_cfg = partial_fields(CropConfig, ArgumentConfig.__dict__)
        warper = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)

        motion_extractor = None
        optimizer_me = None
    elif False:
        warper = _WarpingNetwork(
            num_kp = 64,
            block_expansion = 8,
            max_features = 512,
            num_down_blocks = 2,
            reshape_channel = 32,
            estimate_occlusion_map = False,
            dense_motion_params={
                'block_expansion': 64,
                'max_features': 1024,
                'num_blocks': 2,
                'reshape_depth': 32,
                'compress': 8,
                'down_scale': 1,
            }
        )

        motion_extractor = MotionExtractor(
            num_kp=192,
            backbone='convnextv2_tiny'
        )
        optimizer_me = torch.optim.Adam(motion_extractor.parameters(), lr=opt.lr0, betas=(0.9, 0.99), eps=1e-15)
    elif False:
        warper = WarpingNetwork(
            xf_width=256,
            model_channels=32,
            out_channels=opt.triplane_channels,
            num_res_blocks=1,
            attention_resolutions=[32, 64],
            dropout=0.1,
            channel_mult=(1, 2, 2, 2, 4, 4, 4),
            use_fp16=False,
            num_heads=1,
            num_heads_upsample=-1,
            num_head_channels=8,
            use_scale_shift_norm=False,
            resblock_updown=False,
            in_channels=opt.triplane_channels,
            use_mask=True,
        )

        discriminator = Discriminator(input_channels = 8)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8)
    elif False:
        warper = WarpingNetwork(
            xf_width=256,
            model_channels=32,
            out_channels=2,
            num_res_blocks=2, # 0c 1
            attention_resolutions=[32, 64],
            dropout=0.1,
            channel_mult=(1, 2, 4, 4, 8, 8, 16), # 0c (1, 2, 2, 2, 4, 4, 4)
            use_fp16=False,
            num_heads=1,
            num_heads_upsample=-1,
            num_head_channels=8,
            use_scale_shift_norm=False,
            resblock_updown=False,
            in_channels=2,
            use_mask=False,
        )
    else:
        warper = LightWarpingNetwork(
            in_channels=8,
            scale = 3,
            image_size=512,
            n_feats=8,
            condition_channels=8,
            use_fp16=False,
            use_checkpoint=True,
            ch_mult=[2, 4],
            use_scale_shift_norm=False,
            dtype='32',
            use_3d_conv=False,
            n_resblocks=2,
            latent_type="emo",
        )

    criterion = nn.MSELoss(reduction='mean')
    if optimizer_d:
        optimizer = torch.optim.Adam(warper.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(warper.parameters(), lr=opt.lr0, betas=(0.9, 0.99), eps=1e-8)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1 ** min(iter / opt.num_epochs, 1))
    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()
    trainer = Trainer('snapshot_3', opt, None, warper, motion_extractor, discriminator, local_rank=shard, world_size=num_shards, device=device, workspace=opt.workspace, optimizer=optimizer, optimizer_me=optimizer_me, optimizer_d=optimizer_d, criterion=criterion, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=False, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, renderer_checkpoint=opt.r_ckpt, eval_interval=opt.eval_freq)

    if opt.debug and False:
        x1 = trainer.prepare_source(np.ones((1, 1024, 1024, 3)))
        x2 = trainer.prepare_source(np.ones((1, 1024, 1024, 3))) # 1, 3, 1024, 1024
        t1 = torch.ones(1, 3, 1, 32, 512, 512).to(device)
        t2 = torch.ones(1, 3, 1, 32, 512, 512).to(device)
        trainer.train_step(x1, x2, t1, t2)
    elif opt.debug and False:
        t1 = torch.ones(1, 32, 256, 256 * 3).to(device)
        t2 = torch.ones(1, 32, 256, 256 * 3).to(device)
        r = [torch.ones(1, 128, 64, 64).to(device),
             torch.ones(1, 256, 32, 32).to(device),
             torch.ones(1, 512, 16, 16).to(device)]
        trainer.train_step(t1, r, t2)
    elif opt.debug:
        t1 = torch.ones(1, 32, 512, 512 * 3).to(device)
        t2 = torch.ones(1, 32, 512, 512 * 3).to(device)
        r = torch.ones(1, 4, 16, 16).to(device)
        trainer.train_step(t1, r, t2)


    with open(opt.path, 'r') as f:
        if opt.start_idx >= 0 and opt.end_idx >= 0 and opt.start_idx < opt.end_idx:
            all_ids = f.read().splitlines()[opt.start_idx:opt.end_idx]
        else:
            all_ids = f.read().splitlines()

    # train_loader = TriplaneDataset(opt.src_root, opt.src_data, opt.tgt_root, opt.tgt_data, all_ids, device, batch_size=opt.batch_size, local_rank=shard, world_size=num_shards).dataloader()
    # train_loader = TriplaneImageDataset(opt, opt.src_root, opt.tgt_data, opt.latent_root, resolution=opt.resolution0, all_ids=all_ids, device=device).dataloader()

    train_loader = TriplaneLatentDataset(opt.src_root, opt.tgt_root, opt.tgt_data, opt.latent_root, latent_type=opt.latent_type, triplane_channel=opt.triplane_channels, resolution=opt.resolution0, all_ids=all_ids, device=device, batch_size=opt.batch_size, local_rank=shard, world_size=num_shards).dataloader()
    # train_loader = TriplaneFeatureDataset(opt.src_root, opt.tgt_root, opt.tgt_data, opt.ms_feature_root, preload_mm=True, triplane_channels=opt.triplane_channels, resolution=opt.resolution0, all_ids=all_ids, device=device, batch_size=opt.batch_size, local_rank=shard, world_size=num_shards).dataloader()

    # train_loader = TriplaneImagesDataset(opt, opt.src_root, opt.src_data, opt.tgt_data, all_ids, device, opt.resolution0, local_rank=shard, world_size=num_shards).dataloader()


    trainer.train(train_loader, max_epochs=opt.num_epochs)