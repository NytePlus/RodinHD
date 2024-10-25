from nerf.utils import *
from nerf.utils import Trainer as _Trainer

class Trainer(_Trainer):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer_mlp=None, # optimizer
                 optimizer_triplane=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 lr_scheduler_mlp=None,
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
                 random_noise=False, # whether to add random noise to the input,
                 random_scale=False, # whether to randomly scale the input
                 restore=False, # whether to restore output image
                 ):

        self.opt = opt
        self.lr_scheduler_fn = lr_scheduler
        self.random_noise = random_noise
        self.random_scale = random_scale

        super().__init__(name, opt, model, criterion, optimizer_mlp, optimizer_triplane, ema_decay, lr_scheduler, lr_scheduler_mlp, metrics, local_rank, world_size, device, mute, fp16, eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step, restore)
        
    ### ------------------------------	

    def train_step(self, triplane, data, iwc_state):

        pred_rgb, gt_rgb, loss = super().train_step(triplane, data)

        # l1 reg
        loss += self.model.density_loss(triplane) * self.opt.l1_reg_weight
        loss += self.model.tv_loss(triplane) * self.opt.tv_weight
        loss += self.model.dist_loss(triplane) * self.opt.dist_weight
        if len(iwc_state['fisher_state']) != 0 and len(iwc_state['optpar_state']) != 0:
            loss += self.model.iwc_loss(iwc_state) * self.opt.iwc_weight
        return pred_rgb, gt_rgb, loss

    def train_one_epoch(self, triplane_, loader, iwc_state):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer_mlp.param_groups[0]['lr']:.6f} ..., lr={self.optimizer_triplane.param_groups[0]['lr']:.6f} ...")
        start_time = time.time()

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        # if self.world_size > 1:
        #     loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0 and not self.opt.no_tqdm:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            channels, orig_size = triplane_.shape[-3], triplane_.shape[-1]

            triplane = triplane_.view(-1, channels, orig_size, orig_size)
            triplane = triplane.clamp(-1, 1)
            if self.random_scale:
                random_size = np.random.randint(50, 512)
                triplane = torch.nn.functional.interpolate(triplane, (random_size, random_size), mode='bicubic')
                triplane = torch.nn.functional.interpolate(triplane, (orig_size, orig_size), mode='bicubic')

            if self.random_noise:
                triplane = triplane + 0.001 * torch.randn_like(triplane)

            triplane = triplane.reshape(3, 1, channels, orig_size, orig_size)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state(triplane)
            
            self.local_step += 1
            self.global_step += 1
            
            self.optimizer_mlp.zero_grad()
            self.optimizer_triplane.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(triplane, data, iwc_state)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_mlp)
            self.scaler.step(self.optimizer_triplane)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler_mlp.step()
                self.lr_scheduler_triplane.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
           
                if not self.opt.no_tqdm:
                    if self.scheduler_update_every_step:
                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer_mlp.param_groups[0]['lr']:.6f} , lr={self.optimizer_triplane.param_groups[0]['lr']:.6f}")
                    else:
                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)
        # 每次multiview image的顺序不一样。但是同一个image的光线不会变化

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            if not self.opt.no_tqdm:
                pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler_triplane.step(average_loss)
                self.lr_scheduler_mlp.step(average_loss)
            else:
                self.lr_scheduler_mlp.step()
                self.lr_scheduler_triplane.step()

        epoch_time = time.time() - start_time
        self.log(f"==> Finished {loader._data.subject_id} of Epoch {self.epoch}  | avg loss {average_loss:.4f} | time {epoch_time:.2f} s. Tri../u..")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # mark untrained grid
            if self.global_step == 0:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
                self.error_map = train_loader._data.error_map

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

  

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs


    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}.pth'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'resolution': self.model.resolution, # Different from _Trainer!
        }

        
        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            # state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

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

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        from collections import OrderedDict
        loading_dict = OrderedDict()
        for k, v in checkpoint_dict['model'].items():
            if not 'density_grid' in k and not 'density_bitfield' in k:
                loading_dict[k] = v
 
        # missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        missing_keys, unexpected_keys = self.model.load_state_dict(loading_dict, strict=False)

        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

# Random shuffle rays Trainer by Nyte.
class TrainerPlus(Trainer):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer_mlp=None,  # optimizer
                 optimizers_triplane=None,  # optimizers by Nyte
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 lr_scheduler_mlp=None,
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 random_noise=False,  # whether to add random noise to the input,
                 random_scale=False,  # whether to randomly scale the input
                 restore=False,  # whether to restore output image
                 batch_size=4096,   # batch size
                 ):

        super().__init__(name, opt, model, criterion, optimizer_mlp, None, ema_decay,
                         None, lr_scheduler_mlp, metrics, local_rank, world_size, device, mute, fp16,
                         eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train,
                         use_checkpoint, use_tensorboardX, scheduler_update_every_step, random_noise, random_scale, restore)
        self.optimizers_triplane = optimizers_triplane
        self.batch_size = batch_size

        if lr_scheduler_mlp is not None:  # add by Nyte
            self.lr_scheduler_mlp = lr_scheduler_mlp
            self.lr_schedulers_triplane = [lr_scheduler(optimizer_triplane)
                                           for optimizer_triplane in self.optimizers_triplane]

    # random shuffle rays modified by Nyte.
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch(random shuffle rays) {self.epoch}, lr={self.optimizer_mlp.param_groups[0]['lr']:.6f} ...")
        start_time = time.time()

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.local_rank == 0 and not self.opt.no_tqdm:
            pbar = tqdm.tqdm(total=len(loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            if self.random_scale:
                raise NotImplementedError

            if self.random_noise:
                raise NotImplementedError

            # Deactivate update by Nyte.
            # update grid every 16 steps
            # if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
            #     with torch.cuda.amp.autocast(enabled=self.fp16):
            #         self.model.update_extra_state(triplane)

            self.local_step += 1
            self.global_step += 1

            self.optimizer_mlp.zero_grad()
            for optimizer_triplane in self.optimizers_triplane:
                optimizer_triplane.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgb, gt_rgb, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_mlp)
            for optimizer_triplane in self.optimizers_triplane:
                self.scaler.step(optimizer_triplane)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler_mlp.step()
                for lr_scheduler_triplane in self.lr_schedulers_triplane:
                    lr_scheduler_triplane.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)

                if not self.opt.no_tqdm:
                    if self.scheduler_update_every_step:
                        lr0 = ([optimizer_triplane.param_groups[0]['lr'] for optimizer_triplane in self.optimizers_triplane])
                        lr0_max, lr0_min = max(lr0), min(lr0)
                        pbar.set_description(
                            f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer_mlp.param_groups[0]['lr']:.6f} , lr=[{lr0_min:.6f}, {lr0_max:.6f}]")
                    else:
                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(1)

        # Deactivate all regularization by Nyte.
        # for triplane in triplanes:
        #     # l1 reg
        #     loss += self.model.density_loss(triplane) * self.opt.l1_reg_weight
        #     loss += self.model.tv_loss(triplane) * self.opt.tv_weight
        #     loss += self.model.dist_loss(triplane) * self.opt.dist_weight
        # for iwc_state in iwc_states:
        #     if len(iwc_state['fisher_state']) != 0 and len(iwc_state['optpar_state']) != 0:
        #         loss += self.model.iwc_loss(iwc_state) * self.opt.iwc_weight

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            if not self.opt.no_tqdm:
                pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                for lr_scheduler_triplane in self.lr_schedulers_triplane:
                    lr_scheduler_triplane.step(average_loss)
                self.lr_scheduler_mlp.step(average_loss)
            else:
                self.lr_scheduler_mlp.step()
                for lr_scheduler_triplane in self.lr_schedulers_triplane:
                    lr_scheduler_triplane.step()

        epoch_time = time.time() - start_time
        self.log(
            f"==> Finished all rays of Epoch {self.epoch}  | avg loss {average_loss:.4f} | time {epoch_time:.2f} s.")

    # random shuffle rays modified by Nyte.
    def train_step(self, data):
        rays_o, rays_d, rgbs, triplanes = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in data]
        # print(f'{type(triplanes)} {len(triplanes)} {triplanes[0] is triplanes[1]}')
        # print(f'rays_o: {rays_o.shape} rays_d: {rays_d.shape} triplane: {len(triplanes)}')

        B, C = rgbs.shape

        if self.opt.color_space == 'linear':
            raise NotImplementedError

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        else:
            bg_color = torch.ones(3, device=self.device)

        if C == 4:
            gt_rgbs = rgbs[..., :3] * rgbs[..., 3:] + bg_color * (1 - rgbs[..., 3:])
        else:
            gt_rgbs = rgbs

        # change mean_count by Nyte.
        outputs = self.model(triplanes, rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True,
                             mean_count=rays_o.shape[0] * self.opt.max_steps,
                              force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        pred_rgbs = outputs['image']

        # MSE loss
        loss = self.criterion(pred_rgbs, gt_rgbs).mean(-1)  # [B, 3] --> [B]
        # print(f'pred_rgbs: {pred_rgbs.shape} loss: {loss.shape}')

        # patch-based rendering
        if self.opt.patch_size > 1:
            raise NotImplementedError

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 2:  # [K, B]
            loss = loss.mean(0)

        # Remove update error_map by Nyte.
        # if self.error_map is not None:
        #     index = data['index']  # [B]
        #     inds = data['inds_coarse']  # [B, N]
        #
        #     # take out, this is an advanced indexing and the copy is unavoidable.
        #     error_map = self.error_map[index]  # [B, H * W]
        #
        #     # [debug] uncomment to save and visualize error map
        #     # if self.global_step % 1001 == 0:
        #     #     tmp = error_map[0].view(128, 128).cpu().numpy()
        #     #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
        #     #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        #     #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))
        #
        #     error = loss.detach().to(error_map.device)  # [B, N], already in [0, 1]
        #
        #     # ema update
        #     ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
        #     error_map.scatter_(1, inds, ema_error)
        #
        #     # put back
        #     self.error_map[index] = error_map

        loss = loss.mean()

        return pred_rgbs, gt_rgbs, loss

    # random shuffle rays modified by Nyte.
    def train(self, train_loaders, valid_loader, valid_triplane, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        # if self.model.cuda_ray:
        #     self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loaders)

            if self.epoch % self.eval_interval == 0:
                if valid_loader is not None:
                    self.evaluate_one_epoch(valid_triplane, valid_loader)
                if self.local_rank == 0:
                    self.save_checkpoint(full=False, best=False)

        if valid_loader is not None:
            self.evaluate_one_epoch(valid_triplane, valid_loader)
        if self.local_rank == 0:
            self.save_checkpoint(full=False, best=False)

        # Deactivate all the regulation by Nyte.
        # decoder_state = {}
        # decoder_state.update(self.model.named_parameters())
        # fisher_state = {}
        # optpar_state = {}
        # for name, param in decoder_state.items():
        #     if ('color_net' in name or 'sigma_net' in name) and param.grad is not None:
        #         fisher_state[name] = param.grad.data.clone().pow(2)
        #         optpar_state[name] = param.data.clone()
        # iwc_state["fisher_state"] = [fisher_state]
        # iwc_state["optpar_state"] = [optpar_state]
        # del decoder_state, fisher_state, optpar_state

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()