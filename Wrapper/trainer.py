import contextlib

import torch
import time
import tqdm
import os
import cv2
import random
import tensorboardX

import numpy as np

from torch import nn, optim
from rich.console import Console

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Trainer(object):
    def __init__(self,
                 name, # name of this experiment
                 opt, # extra conf
                 renderer, # triplane renderer
                 wrapper, # wrapping module
                 motion_extractor, # motion extractor
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 lr_scheduler=None, # scheduler
                 optimizer=None, # wrapping module optimizer
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.optimizer = optimizer

        renderer.to(self.device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dict = torch.load(current_dir + '/weights/renderer.pth', map_location=self.device)
        from collections import OrderedDict
        loading_dict = OrderedDict()
        for k, v in checkpoint_dict['model'].items():
            if not 'density_grid' in k and not 'density_bitfield' in k:
                loading_dict[k] = v
        renderer.load_state_dict(loading_dict, strict=False)
        self.renderer = renderer

        wrapper.to(self.device)
        self.wrapper = wrapper

        motion_extractor.to(self.device)
        motion_extractor.load_pretrained(current_dir + '/weights/motion_extractor.pth')
        self.motion_extractor = motion_extractor

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in wrapper.parameters() if p.requires_grad])}')

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                if self.opt.no_tqdm:
                    print(*args)
                else:
                    self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def eval_step(self, triplane, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        images = data['images']  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.renderer(triplane, rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    def evaluate(self, triplane, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        psnr = self.evaluate_one_epoch(triplane, loader, name)
        self.use_tensorboardX = use_tensorboardX
        return psnr

    def evaluate_one_epoch(self, triplane, loader, name=None):
        if self.local_rank == 0:
            self.log(f"++> Evaluate {loader._data.subject_id} at epoch {self.epoch} ...")

            if name is None:
                name = f'{self.name}_ep{self.epoch:04d}'

            total_loss = 0
            for metric in self.metrics:
                metric.clear()

            self.wrapper.eval()

            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            with torch.no_grad():
                self.local_step = 0

                for data in loader:
                    self.local_step += 1

                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, truths, loss = self.eval_step(triplane, data)

                    loss_val = loss.item()
                    total_loss += loss_val

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{loader._data.subject_id}',
                                             f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{loader._data.subject_id}',
                                                   f'{name}_{self.local_step:04d}_depth.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

            average_loss = total_loss / self.local_step
            self.stats["valid_loss"].append(average_loss)

            psnr = self.metrics[0].measure()
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report() + f" {loader._data.subject_id}", style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

            self.log(f"++> Evaluate {loader._data.subject_id} of epoch {self.epoch} Finished.")
            return psnr

    def inference_ctx(self):
        if self.device == "mps":
            ctx = contextlib.nullcontext()
        else:
            ctx = torch.autocast(device_type=self.device[:4], dtype=torch.float16,
                                 enabled=self.inference_cfg.flag_use_half_precision)
        return ctx

    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        with torch.no_grad(), self.inference_ctx():
            kp_info = self.motion_extractor(x)

            if self.fp16:
                # float the dict
                for k, v in kp_info.items():
                    if isinstance(v, torch.Tensor):
                        kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        h, w = img.shape[:2]
        if h != self.inference_cfg.input_shape[0] or w != self.inference_cfg.input_shape[1]:
            x = cv2.resize(img, (self.inference_cfg.input_shape[0], self.inference_cfg.input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(self.device)
        return x

    def train_step(self, ref_photo, src_photo, src_triplane, tgt_triplane):
        ref_kp = self.get_kp_info(ref_photo) # BxNx3
        src_kp = self.get_kp_info(src_photo) # BxNx3

        wrapped_triplane = self.wrapper(src_triplane, ref_kp, src_kp) # B×3×1×32×512×512

        # MSE loss
        loss = self.criterion(src_triplane, tgt_triplane).mean(-1) # [B, N, 3] --> [B, N]
        loss = loss.mean()

        return loss, wrapped_triplane

    def train_one_epoch(self, triplane_, loader):
        self.log(f"Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.wrapper.train()
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            triplane = triplane_.view(-1, 96, 512, 512)
            triplane = triplane.clamp(-1, 1)

            triplane = triplane.reshape(3, 1, 32, 512, 512)

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss, wapped_tp = self.train_step(triplane, data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar(f'train/loss', loss_val, self.global_step)
                    self.writer.add_scalar(f'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def train(self, train_loader, valid_loader, max_epochs, triplane):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.evaluate_one_epoch(triplane, valid_loader)

            self.train_one_epoch(triplane, train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(triplane, valid_loader)
                if self.local_rank == 0:
                    self.save_checkpoint(full=False, best=False)

        self.evaluate_one_epoch(triplane, valid_loader)
        if self.local_rank == 0:
            self.save_checkpoint(full=False, best=False)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}.pth'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'resolution': self.renderer.resolution,  # Different from _Trainer!
        }

        if self.renderer.cuda_ray:
            state['mean_count'] = self.renderer.mean_count
            state['mean_density'] = self.renderer.mean_density

        if full:
            # state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()

        if not best:

            state['model'] = self.wrapper.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    state['model'] = self.wrapper.state_dict()
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")