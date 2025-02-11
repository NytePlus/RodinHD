import os
import sys
import torch
import argparse
import numpy as np

sys.path.append('../')

from Renderer import dist_util
from torch import optim, nn
from mpi4py import MPI
from Renderer.TriplaneFit.network import NeRFNetwork

from Renderer.nerf.provider import NeRFDataset
from Warper.metrics import PSNRMeter
from Warper.modules.motion_extractor import MotionExtractor
from Warper.trainer import seed_everything, Trainer, TriplaneDataset
from Warper.modules.warping_network import WarpingNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--src_root', type=str, default='')
    parser.add_argument('--tgt_root', type=str, default='')
    parser.add_argument('--src_data', type=str, default='')
    parser.add_argument('--tgt_data', type=str, default='')
    parser.add_argument('-debug', action='store_true', help="debug for tensor dims")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='facescape')
    ### training options
    parser.add_argument('--num_epochs', type=int, default=9000, help="training iters")
    parser.add_argument('--lr0', type=float, default=2e-2, help="initial learning rate for warper")
    parser.add_argument('--ckpt', type=str, default='')

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--cp', action='store_true', help="use TensorCP")
    parser.add_argument('--triplane_channels', type=int, default=32)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--resolution0', type=int, default=512)
    parser.add_argument('--resolution1', type=int, default=512)

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

    opt = parser.parse_args()
    seed_everything(opt.seed)

    dist_util.setup_dist()
    print('setup done.')
    device = dist_util.dev()

    warper = WarpingNetwork(
        num_kp = 21,
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
            'down_scale': 2,
        }
    )

    motion_extractor = MotionExtractor(
        num_kp = 21,
        backbone = 'convnextv2_tiny'
    )

    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(warper.parameters(), lr=opt.lr0, betas=(0.9, 0.99), eps=1e-15)
    optimizer_me = torch.optim.Adam(motion_extractor.parameters(), lr=opt.lr0, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.01 ** min(iter / opt.num_epochs, 1))
    trainer = Trainer('snapshot_3', opt, warper, motion_extractor, local_rank=shard, world_size=num_shards, device=device, workspace=opt.workspace, optimizer=optimizer, optimizer_me=optimizer_me, criterion=criterion, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=False, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=opt.eval_freq)

    if opt.debug:
        x1 = trainer.prepare_source(np.ones((1, 1024, 1024, 3)))
        x2 = trainer.prepare_source(np.ones((1, 1024, 1024, 3))) # 1, 3, 1024, 1024
        t1 = torch.ones(1, 3, 1, 32, 512, 512).to(device)
        t2 = torch.ones(1, 3, 1, 32, 512, 512).to(device)
        trainer.train_step(x1, x2, t1, t2)

    with open(opt.path, 'r') as f:
        if opt.start_idx >= 0 and opt.end_idx >= 0 and opt.start_idx < opt.end_idx:
            all_files = f.read().splitlines()[opt.start_idx:opt.end_idx]
        else:
            all_files = f.read().splitlines()
    all_ids = all_files

    train_loader = TriplaneDataset(src_root=opt.src_root, src_data=opt.src_data,
                                   tgt_root=opt.tgt_root, tgt_data=opt.tgt_data, all_ids=all_ids,
                                   local_rank=shard, num_shards=num_shards,
                                   batch_size=opt.batch_size, device=device).dataloader()

    trainer.train(train_loader, max_epochs=opt.num_epochs)