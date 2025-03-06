import contextlib

import torch
import time
import tqdm
import glob
import os
import cv2
import random
import tensorboardX

import numpy as np

from torch import nn, optim
from rich.console import Console

from nerf.utils import Trainer as _Trainer
from pkg_resources import parse_version

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

class Trainer(_Trainer):
    def __init__(self,
                 name, # name of this experiment
                 opt, # extra conf
                 renderer, # nerf renderer
                 warper, # warping module
                 motion_extractor, # motion extractor
                 discriminator, # gan discriminator
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 lr_scheduler=None, # scheduler
                 optimizer=None, # wrapping module optimizer
                 optimizer_me=None, # motion extracter optimizer
                 optimizer_d=None, # discriminator optimizer
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
                 renderer_checkpoint="",
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 input_shape=[256,256], # input shape of source image
                 ):
        super().__init__(name, opt, renderer, criterion, fp16=fp16, use_checkpoint = renderer_checkpoint, local_rank = local_rank, world_size = world_size, device = device, workspace=workspace)

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
        self.optimizer_me = optimizer_me
        self.optimizer_d = optimizer_d
        self.scheduler_fn = lr_scheduler
        self.input_shape = input_shape

        warper.to(self.device)
        if self.world_size > 1:
            warper = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warper)
            self._warper = torch.nn.parallel.DistributedDataParallel(warper, device_ids=[local_rank],
                                                                    find_unused_parameters=False)
            self.warper = self._warper.module
        else:
            self.warper = self._warper = warper

        self.discriminator = discriminator
        if discriminator is not None:
            discriminator.to(self.device)
            if self.world_size > 1:
                discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
                self._discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank],
                                                                         find_unused_parameters=False)
                self.discriminator = self._discriminator.module
            else:
                self.discriminator = self._discriminator = discriminator

            self.adversarial_loss = nn.BCEWithLogitsLoss()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.motion_extractor = motion_extractor
        if motion_extractor is not None:
            motion_extractor.to(self.device)

            self.scaler_me = torch.cuda.amp.GradScaler(enabled=self.fp16)
            if self.world_size > 1:
                motion_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(motion_extractor)
                self._motion_extractor = torch.nn.parallel.DistributedDataParallel(motion_extractor, device_ids=[local_rank],
                                                                         find_unused_parameters=False)
                self.motion_extractor = self._motion_extractor.module
            else:
                self.motion_extractor = self._motion_extractor = motion_extractor
            motion_extractor.load_pretrained(current_dir + '/weights/motion_extractor.pth', device)

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

            self.valid_path = os.path.join(self.workspace, f'validation/{self.name}')
            os.makedirs(self.valid_path, exist_ok=True)

        self.log(f'[INFO] #parameters: {sum([p.numel() for p in warper.parameters() if p.requires_grad])}')

        self.epoch = 0
        self.local_step = 0
        self.global_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        if use_checkpoint == "latest":
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                state_dict = torch.load(checkpoint_list[-1])
                ret = self.warper.load_state_dict(state_dict["model"], strict=True)
                ret2 = self.discriminator.load_state_dict(state_dict["model_d"], strict=True) if discriminator else None
                self.global_step = state_dict['global_step']
                self.log(f"[INFO] Latest checkpoint is {checkpoint_list[-1]}, ret: {ret} {ret2}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")

        if parse_version(torch.__version__) < parse_version("1.14.0"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        else:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.fp16)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(optimizer)

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        self.error_map = None

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

    def evaluate_one_epoch(self, loader, name=None):
        if self.local_rank == 0:
            self.log(f"++> Evaluate {loader._data.subject_id} at epoch {self.epoch} ...")

            if name is None:
                name = f'{self.name}_ep{self.epoch:04d}'

            total_loss = 0
            for metric in self.metrics:
                metric.clear()

            self.warper.eval()

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
                    save_path = os.path.join(self.workspace, f'validation/{self.name}', f'{loader._data.subject_id}',
                                             f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, f'validation/{self.name}', f'{loader._data.subject_id}',
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
            ctx = torch.autocast('cuda', dtype=torch.float16,
                                 enabled=self.fp16)
        return ctx

    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        kp_info = self.motion_extractor(x)

        if self.fp16:
            # float the dict
            for k, v in kp_info.items():
                if isinstance(v, torch.Tensor):
                    kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs, num_kpx2 = kp_info['kp_2d'].shape
            # kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)
            kp_info['kp_2d'] = torch.stack(torch.split(kp_info['kp_2d'].reshape(bs, -1, 2), num_kpx2 // 6, dim=1), dim=0)

        return kp_info

    def prepare_source(self, img) -> torch.Tensor:
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        if len(img.shape) == 4:
            h, w = img.shape[1:3]
        else:
            h, w = img.shape[:2]
        if h != self.input_shape[0] or w != self.input_shape[1]:
            if len(img.shape) == 4:
                x = np.array([cv2.resize(i, (self.input_shape[0], self.input_shape[1])) for i in img])
            else:
                x = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
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

    def liveportrait_train_step(self, src_photo, ref_photo, src_triplane, tgt_triplane):
        warpped_triplane = self.warper.execute(src_triplane, src_photo, ref_photo)

        loss = self.criterion(warpped_triplane, tgt_triplane).mean()
        ref_loss = self.criterion(src_triplane, tgt_triplane).mean()

        return warpped_triplane, loss, ref_loss

    def end2end_train_step(self, src_photo, ref_photo, src_triplane, tgt_triplane):
        bs, _, _, c, h, w = src_triplane.shape

        ref_kp = self.get_kp_info(ref_photo)['kp_2d'] # BxNx3
        src_kp = self.get_kp_info(src_photo)['kp_2d'] # BxNx3

        wrapped_dict = self.warper(src_triplane, ref_kp, src_kp) # B×3×1×32×512×512
        warpped_triplane = wrapped_dict['out'].view(bs, 3, 1, c, h, w)

        # MSE loss
        loss = self.criterion(warpped_triplane, tgt_triplane).mean()
        ref_loss = self.criterion(src_triplane, tgt_triplane).mean()

        return warpped_triplane, loss, ref_loss

    def train_step(self, src_photo, ref_photo, src_triplane, img_loader):
        bs, _3, _1, c, h, w = src_triplane.shape

        ref_kp = self.get_kp_info(ref_photo)['kp_2d']  # BxNx3
        src_kp = self.get_kp_info(src_photo)['kp_2d']  # BxNx3

        for i, data in enumerate(img_loader):
            self.local_step += 1
            self.global_step += 1

            if self.local_rank == 0 and self.report_metric_at_train:
                for metric in self.metrics:
                    metric.clear()

            if parse_version(torch.__version__) >= parse_version("1.14.0"):
                with torch.amp.autocast(device_type='cuda', enabled=self.fp16):
                    wraped_dict = self.warper(src_triplane, ref_kp, src_kp)  # B×3×1×32×512×512
                    warped_triplane = wraped_dict['out'].view(bs, 3, 1, c, h, w)

                    pred_rgb, gt_rgb, loss = super().train_step(warped_triplane, data)
            else:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    wraped_dict = self.warper(src_triplane, ref_kp, src_kp)  # B×3×1×32×512×512
                    warped_triplane = wraped_dict['out'].view(bs, 3, 1, c, h, w)

                    pred_rgb, gt_rgb, loss = super().train_step(warped_triplane, data)

            loss_val = loss.item()
            self.total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(pred_rgb, gt_rgb)

                if self.use_tensorboardX:
                    self.writer.add_scalar(f'train/loss', loss_val, self.global_step)
                    self.writer.add_scalar(f'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    self.pbar.set_description(
                        f"loss={loss_val:.4f} ({self.total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    self.pbar.set_description(f"loss={loss_val:.4f} ({self.total_loss / self.local_step:.4f})")
                self.pbar.update(1)

            self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        average_loss = self.total_loss / self.local_step

        return warped_triplane, average_loss

    def aborted_union_train_step(self, src_triplane, ref_feature, img_loader):
        # TODO:
        #  1. Evaluate
        #  2. optimizer isn't work
        #  ~~ 3. raymarching(when not fp16) and grid_sample(when fp16) appears nan ~~ it's all pytorch version's fault
        self.local_step = 0

        self.optimizer.zero_grad()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(img_loader),
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        total_loss = 0
        for data in img_loader:
            self.local_step += 1
            self.global_step += 1

            if self.local_rank == 0 and self.report_metric_at_train:
                for metric in self.metrics:
                    metric.clear()

            # Nyte: fp16 and Adam eps=1e-15 cause nan.
            with torch.cuda.amp.autocast(enabled=self.fp16):
                warped_triplane = self.warper(src_triplane, ref_feature, src_triplane)
                warped_triplane = warped_triplane.reshape(1, 32, 3, self.opt.resolution0, self.opt.resolution1)
                warped_triplane = warped_triplane.permute(2, 0, 1, 3, 4).contiguous().clamp(-1.0, 1.0)

                pred_rgb, gt_rgb, loss = super().train_step(warped_triplane, data)

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(pred_rgb, gt_rgb)

                if self.use_tensorboardX:
                    self.writer.add_scalar(f'train/loss', loss_val, self.global_step)
                    self.writer.add_scalar(f'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(1)

            self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)

        average_loss = total_loss / self.local_step

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        return warped_triplane, pred_rgb, gt_rgb, average_loss

    def aborted_train_step(self, src_triplane, tgt_triplane, latent):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            delta = self.warper(src_triplane, latent)
            warped_triplane = delta # * mask # (3, 32, 512, 1536)

            # test
            # mask_gt = (torch.norm(tgt_triplane - src_triplane, dim=1) > 0.5).unsqueeze(1).type(src_triplane.dtype)
            # tgt_triplane = tgt_triplane * mask_gt + src_triplane * (1 - mask_gt)
            # mask = (mask > 0.5).type(src_triplane.dtype)
            # warped_triplane = src_triplane + delta * mask
            #
            # loss = self.criterion(warped_triplane * mask_gt, tgt_triplane * mask_gt) + self.criterion(mask, mask_gt)
            #
            # return warped_triplane, loss, self.criterion(src_triplane, tgt_triplane), self.criterion(warped_triplane, tgt_triplane)
        return warped_triplane, self.criterion(warped_triplane, tgt_triplane), self.criterion(src_triplane, tgt_triplane)

    def gan_train_step(self, src_triplane, tgt_triplane, latent, lambd = 0.7):
        torch.autograd.set_detect_anomaly(True)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            delta, mask = self.warper(src_triplane, latent)
            warped_triplane = delta * mask # (bs, 32, 512, 1536)

            real_labels = torch.ones(src_triplane.shape[0], 1).to(self.device)
            fake_labels = torch.zeros(src_triplane.shape[0], 1).to(self.device)

        for _ in range(15):
            real_loss = self.adversarial_loss(self.discriminator(tgt_triplane), real_labels)
            fake_loss = self.adversarial_loss(self.discriminator(warped_triplane.detach()), fake_labels)

            d_loss = (real_loss + fake_loss) / 2

            self.optimizer_d.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_d)
            self.scaler.update()

        mse_loss = self.criterion(warped_triplane, tgt_triplane)
        loss = self.adversarial_loss(self.discriminator(warped_triplane), real_labels) + lambd * mse_loss
        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return warped_triplane, mse_loss, d_loss

    def train_one_epoch(self, loader):
        self.local_step = 0

        total_loss = 0
        reftotal_loss = 0
        self.log(f"Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader),
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.warper.train()
        if self.motion_extractor:
            self.motion_extractor.train()
        local_batch_size = 0

        # for i, (src_x_np, ref_x_np, src_t, tgt_t, names) in enumerate(loader):
        for i, (src_t, tgt_t, latent, names) in enumerate(loader):
            self.local_step += 1
            self.global_step += 1

            if self.local_rank == 0 and self.report_metric_at_train:
                for metric in self.metrics:
                    metric.clear()

            self.optimizer.zero_grad()
            if self.optimizer_me:
                self.optimizer_me.zero_grad()
            # src_x = self.prepare_source(src_x_np).to(self.device)
            # ref_x = self.prepare_source(ref_x_np).to(self.device)

            # union
            src_t = src_t.to(self.device)
            tgt_t = tgt_t.to(self.device)
            if isinstance(latent, (tuple, list)):
                latent = [ls.to(self.device) for ls in latent]
            else:
                latent = latent.to(self.device)

            # with torch.cuda.amp.autocast(enabled=self.fp16):
            #     warped_t, loss, ref_loss = self.end2end_train_step(src_x, ref_x, src_t, tgt_t)
            # with torch.cuda.amp.autocast(enabled=self.fp16):
            #     warped_t, loss, ref_loss = self.liveportrait_train_step(src_x, ref_x, src_t, tgt_t)
            with torch.cuda.amp.autocast(enabled=self.fp16):
                warped_t, loss, ref_loss = self.aborted_train_step(src_t, tgt_t, latent)
                # warped_t, loss, ref_loss = self.gan_train_step(src_t, tgt_t, latent)

            if self.global_step % 1000 == i:
                for t, n in zip(warped_t, names):
                    with open(os.path.join(self.workspace, f'validation/{self.name}', n), 'wb') as f:
                        np.save(f, torch.stack(t.split(t.shape[-2], dim = -1)).detach().cpu().numpy())

            if self.optimizer_d is None:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.warper.parameters(), max_norm=5.0)

                # if self.local_rank == 0:
                #     for name, param in self.warper.named_parameters():
                #         if param.grad is not None:
                #             print(f"Layer: {name} | {param.type()} gradient: [{param.grad.min().item()}, {param.grad.max().item()}]")

                self.scaler.step(self.optimizer)
                if self.optimizer_me:
                    torch.nn.utils.clip_grad_norm_(self.motion_extractor.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer_me)
                self.scaler.update()

            loss_val = loss.item()
            total_loss += loss_val

            reftotal_loss += ref_loss.item()
            refavg_loss = reftotal_loss / self.local_step

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(warped_t, tgt_t)

                if self.use_tensorboardX:
                    self.writer.add_scalar(f'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}) ref={refavg_loss:.4f}")
                pbar.update(1)


            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

        average_loss = total_loss / self.local_step

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

            if self.use_tensorboardX:
                self.writer.add_scalar(f'train/loss', average_loss, self.global_step)

        self.stats["loss"].append(average_loss)

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def aborted_train_one_epoch(self, loader):
        self.local_step = 0

        self.total_loss = 0
        self.log(f"Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        if self.local_rank == 0:
            self.pbar = tqdm.tqdm(total=len(loader) * len(loader.dataset[0][3]),
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.warper.train()

        for src_t, src_x_np, ref_x_np, img_loader, tgt_ids in loader:
            src_x = self.prepare_source(src_x_np).to(self.device)
            ref_x = self.prepare_source(ref_x_np).to(self.device)
            src_t = src_t.to(self.device)


            if parse_version(torch.__version__) >= parse_version("1.14.0"):
                with torch.amp.autocast(device_type='cuda', enabled=self.fp16):
                    warped_t, average_loss = self.train_step(src_x, ref_x, src_t, img_loader)
            else:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    warped_t, average_loss = self.train_step(src_x, ref_x, src_t, img_loader)

            for t, n in zip(warped_t, tgt_ids):
                with open(os.path.join(self.workspace, 'validation', n), 'wb') as f:
                    np.save(f, t.detach().cpu().numpy())

        if self.local_rank == 0:
            self.pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

            if self.use_tensorboardX:
                self.writer.add_scalar(f'train/loss', average_loss, self.global_step)

        self.stats["loss"].append(average_loss)

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def train(self, train_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                # self.evaluate_one_epoch(train_loader)
                if self.local_rank == 0:
                    self.save_checkpoint(full=False, best=False)

        # self.evaluate_one_epoch(triplane, valid_loader)
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
        }


        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()

        if not best:

            state['model'] = self.warper.state_dict()
            if self.discriminator:
                state['model_d'] = self.discriminator.state_dict()

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

                    state['model'] = self.warper.state_dict()
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
