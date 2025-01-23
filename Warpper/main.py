import os
import torch
import argparse

from torch import optim, nn
from Renderer.TriplaneFit.network import NeRFNetwork

from Renderer.nerf.provider import NeRFDataset
from Warpper.metrics import PSNRMeter
from Warpper.modules.motion_extractor import MotionExtractor
from Warpper.trainer import seed_everything, Trainer
from Warpper.modules.warping_network import WarpingNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--ray_shuffle', action='store_true',
                        help="Nyte's modify. train random rays instead of one avatar.")
    parser.add_argument('--no_grid', action='store_true', help="Nyte's modify. set density grid full.")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='facescape')
    ### training options
    parser.add_argument('--iters', type=int, default=9000, help="training iters")
    parser.add_argument('--lr0', type=float, default=2e-2, help="initial learning rate for embeddings")
    parser.add_argument('--lr1', type=float, default=0, help="initial learning rate for networks")
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--num_rays', type=int, default=8192, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=8192,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--l1_reg_weight', type=float, default=1e-4)
    parser.add_argument('--tv_weight', type=float, default=0.02)
    parser.add_argument('--dist_weight', type=float, default=0.002)
    parser.add_argument('--iwc_weight', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--cp', action='store_true', help="use TensorCP")
    parser.add_argument('--triplane_channels', type=int, default=32)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--resolution0', type=int, default=512)
    parser.add_argument('--resolution1', type=int, default=512)
    parser.add_argument("--upsample_model_steps", type=int, action="append", default=[])
    parser.add_argument('--grid_size', type=int, default=256)

    ### dataset options
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

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--num_train_frames', type=int, default=300, help="num of training frames")
    parser.add_argument('--out_loop_eps', type=int, default=1, help="outloop eps")
    parser.add_argument('--eval_freq', type=int, default=10, help="eval freq")
    parser.add_argument('--no_tqdm', action='store_true', help="disable tqdm")
    parser.add_argument('--random_noise', action='store_true', help="add random noise to triplane")
    parser.add_argument('--random_scale', action='store_true', help="add random scaling to triplane")
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument('--start_idx', type=int, default=-1, help="start idx of index file, <0 uses no start idx")
    parser.add_argument('--end_idx', type=int, default=-1, help="end idx of index file, <0 uses no end idx")
    parser.add_argument('--eval_video', action='store_true', help="eval test video")

    opt = parser.parse_args()
    seed_everything(opt.seed)
    device='cuda:0'
    max_epoch = 300

    renderer = NeRFNetwork(
        resolution=[opt.resolution0] * 3,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        sigma_rank=[int(opt.triplane_channels // 4)] * 3,
        color_rank=[int(opt.triplane_channels // 4 * 3)] * 3,
        triplane_channels=opt.triplane_channels,
    )

    wrapper = WarpingNetwork(
        num_kp = 21,
        block_expansion = 64,
        max_features = 512,
        num_down_blocks = 2,
        reshape_channel = 32,
        estimate_occlusion_map = True,
        dense_motion_params={
            'block_expansion': 32,
            'max_features': 1024,
            'num_blocks': 5,
            'reshape_depth': 16,
            'compress': 4,
        }
    )

    x1 = torch.ones(64, 1024, 1024)
    x2 = torch.ones(64, 1024, 1024)
    t = torch.ones(64, 3, 1, 32, 512, 512)

    motion_extractor = MotionExtractor(
        num_kp = 21,
        backbone = 'convnextv2_tiny'
    )

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=opt.lr0, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    trainer = Trainer('snapshot', opt, renderer, wrapper, motion_extractor, local_rank=0, world_size=1, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=opt.eval_freq)
    

    shard = 0
    num_shards = 1
    with open(opt.path, 'r') as f:
        subject_id = f.read().splitlines()[opt.start_idx:opt.end_idx][shard:][::num_shards]

    for sid in subject_id:
        train_loader, triplane, iwc_state = NeRFDataset(opt, root_path=os.path.join(opt.data_root, sid),
                                                        save_dir=opt.save_dir, device=device, type='train',
                                                        triplane_resolution=opt.resolution0,
                                                        triplane_channels=opt.triplane_channels, downscale=opt.downscale,
                                                        num_train_frames=None).dataloader()
        triplane = triplane.reshape(3, 1, opt.triplane_channels, opt.resolution0, opt.resolution0)
        triplane = triplane.clamp(-1.0, 1.0)
        triplane = triplane.to(device)
        triplane.requires_grad = True

        valid_loader, triplane_ = NeRFDataset(opt, root_path=os.path.join(opt.data_root, sid), save_dir=opt.save_dir,
                                              device=device, type='val', downscale=opt.downscale,
                                              triplane_resolution=opt.resolution0, triplane_channels=opt.triplane_channels,
                                              num_train_frames=opt.num_train_frames).dataloader()

        trainer.train(train_loader, valid_loader, max_epoch, triplane)