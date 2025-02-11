import os
import glob
import torch
import cv2
import numpy as np


from uitls import load_triplane, load_image_rgb
from concurrent.futures.thread import ThreadPoolExecutor

from torch.utils.data import DataLoader, Dataset
from nerf.provider import NeRFDataset

class TriplaneDataset(Dataset):
    def __init__(self,
                 src_root,
                 src_data,
                 tgt_root,
                 tgt_data,
                 all_ids,
                 device,
                 local_rank=0,
                 num_shards=1,
                 batch_size=1,
                 preload_mm=True,
                 ):
        self.src_root = src_root
        self.src_data = src_data
        self.tgt_root = tgt_root
        self.tgt_data = tgt_data
        self.batch_size = batch_size
        self.device = device
        self.preload_mm = preload_mm

        self.all_ids = all_ids
        self.src_photos = []
        self.ref_photos = []
        self.src_tpath = []
        self.tgt_tpath = []
        self.all_pair_ids = []
        self.pair_ids = []
        self.name_dict = {}

        for id in all_ids:
            base_id = os.path.basename(id)
            src_triplane_path = os.path.join(self.src_root, base_id + '.npy')
            self.src_tpath.append(src_triplane_path)

            # NOTE: the first img is front.
            src_photo_path = os.path.join(self.src_data, base_id, 'img_proc_fg_000000.png')
            self.src_photos.append(load_image_rgb(src_photo_path))

            tgt_files = glob.glob(os.path.join(tgt_root, base_id + '_*.npy'))
            for tgt_triplane_path in tgt_files:
                tgt_base_id = os.path.splitext(os.path.basename(tgt_triplane_path))[0]
                self.tgt_tpath.append(tgt_triplane_path)
                self.all_pair_ids.append([len(self.src_tpath) - 1, len(self.tgt_tpath) - 1])

                tgt_photo_path = os.path.join(self.tgt_data, tgt_base_id, 'img_proc_fg_000000.png')
                self.ref_photos.append(load_image_rgb(tgt_photo_path))

        self.src_triplanes = [None] * len(self.src_tpath)
        self.tgt_triplanes = [None] * len(self.tgt_tpath)
        self.pair_ids = self.all_pair_ids[local_rank:][::num_shards]

        if self.preload_mm:
            len_src = len(self.src_tpath)
            with ThreadPoolExecutor() as executor:
                executor.map(self.load_triplane_from_idx, range(len_src), ['src'] * len_src)

            len_tgt = len(self.tgt_tpath)
            with ThreadPoolExecutor() as executor:
                executor.map(self.load_triplane_from_idx, range(len_tgt), ['tgt'] * len_tgt)

    def load_triplane_from_idx(self, idx, type: str):
        if type == 'src':
            triplane = load_triplane(self.src_tpath[idx])
            self.src_triplanes[idx] = triplane
            self.name_dict[triplane] = self.src_tpath[idx].split('/')[-1]
            return triplane, self.name_dict[triplane]
        elif type == 'tgt':
            print(f'loading triplane from {self.tgt_tpath[idx]}')
            triplane = load_triplane(self.tgt_tpath[idx])
            self.tgt_triplanes[idx] = triplane
            self.name_dict[triplane] = self.tgt_tpath[idx].split('/')[-1]
            return triplane, self.name_dict[triplane]
        else:
            raise ValueError(f"Unknown triplane type: {type}")

    def __getitem__(self, idx):
        src_id, tgt_id = self.pair_ids[idx]
        if not self.preload_mm:
            src_t, _ = self.load_triplane_from_idx(src_id, 'src')
            tgt_t, name = self.load_triplane_from_idx(tgt_id, 'tgt')
            return self.src_photos[src_id], self.ref_photos[tgt_id], src_t, tgt_t, name
        else:
            tgt_triplane = self.tgt_triplanes[tgt_id]
            return self.src_photos[src_id], self.ref_photos[tgt_id], \
                self.src_triplanes[src_id], tgt_triplane, self.name_dict[tgt_triplane]

    def __len__(self):
        return len(self.pair_ids)

    def dataloader(self):
        def custom_collate_fn(batch):
            src_x, ref_x, src_t, tgt_t, names = zip(*batch) # tuple
            return np.stack(src_x), np.stack(ref_x), \
                torch.stack(src_t).to(self.device), \
                torch.stack(tgt_t).to(self.device), \
                names
        return DataLoader(self, batch_size = self.batch_size, shuffle = True, collate_fn = custom_collate_fn)

class TriplaneImageDataset(Dataset):
    def __init__(self,
                 opt,
                 src_root,
                 tgt_data,
                 latent_root,
                 all_ids,
                 device,
                 resolution):
        self.src_root = src_root
        self.tgt_data = tgt_data
        self.latent_root = latent_root
        self.all_ids = all_ids
        self.device = device
        self.batch_size = 1
        self.opt = opt
        self.resolution = resolution
        self.all_ids = []

        for id in all_ids:
            base_id = os.path.basename(id)

            tgt_paths = glob.glob(os.path.join(tgt_data, base_id + '_*'))
            for tgt_path in tgt_paths:
                tgt_id = os.path.basename(tgt_path)
                self.all_ids.append([base_id, tgt_id])

        print(len(self.all_ids))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        src_id, tgt_id = self.all_ids[idx]
        train_loader, _, _ = NeRFDataset(self.opt, root_path=os.path.join(self.tgt_data, tgt_id),
                                                        save_dir=self.opt.save_dir, device=self.device, type='train',
                                                        triplane_resolution=self.opt.resolution0,
                                                        triplane_channels=self.opt.triplane_channels,
                                                        downscale=self.opt.downscale, num_train_frames=None).dataloader()
        triplane_path = os.path.join(self.src_root, src_id + '.npy')

        print("Loading triplane from {}".format(triplane_path))
        with open(triplane_path, 'rb') as f:
            triplane = np.load(f)
            triplane = torch.as_tensor(triplane, dtype=torch.float32)
            triplane = triplane.reshape(3, 32, self.resolution, self.resolution)
            triplane = torch.cat([triplane[0], triplane[1], triplane[2]], -1).unsqueeze(0)

        latent = torch.load(os.path.join(self.latent_root, tgt_id + '.pt'))
        mean, logvar = torch.chunk(latent, 2, dim=0)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.rand_like(mean)
        label_tensor = mean + std * sample

        return triplane, mean.unsqueeze(0), train_loader, tgt_id

    def dataloader(self):
        def collate_fn(batch):
            return batch[0]
        return DataLoader(self, batch_size = self.batch_size, shuffle = False, collate_fn = collate_fn)


class TriplaneLatentDataset(Dataset):
    def __init__(self,
                 src_root,
                 tgt_root,
                 tgt_data,
                 latent_root,
                 all_ids,
                 device,
                 resolution,
                 batch_size=1,
                 preload_mm=True):
        self.src_root = src_root
        self.latent_root = latent_root
        self.tgt_root = tgt_root
        self.tgt_data = tgt_data
        self.batch_size = batch_size
        self.device = device
        self.preload_mm = preload_mm
        self.resolution = resolution

        self.all_ids = all_ids
        self.latent_path = []
        self.src_tpath = []
        self.tgt_tpath = []
        self.pair_ids = []

        for id in all_ids:
            base_id = os.path.basename(id)
            src_triplane_path = os.path.join(self.src_root, base_id + '.npy')
            self.src_tpath.append(src_triplane_path)


            tgt_files = glob.glob(os.path.join(tgt_root, base_id + '_*.npy'))
            for tgt_triplane_path in tgt_files:
                self.tgt_tpath.append(tgt_triplane_path)

                tgt_base_id = os.path.splitext(os.path.basename(tgt_triplane_path))[0]
                tgt_photo_path = os.path.join(self.latent_root, tgt_base_id + '.pt')
                self.latent_path.append(tgt_photo_path)

                self.pair_ids.append([len(self.src_tpath) - 1, len(self.tgt_tpath) - 1])

        self.src_triplanes = [None] * len(self.src_tpath)
        self.tgt_triplanes = [None] * len(self.tgt_tpath)
        self.latents = [None] * len(self.latent_path)

        if self.preload_mm:
            len_src = len(self.src_tpath)
            with ThreadPoolExecutor() as executor:
                executor.map(self.load_triplane_from_idx, range(len_src), ['src'] * len_src)

            len_tgt = len(self.tgt_tpath)
            with ThreadPoolExecutor() as executor:
                executor.map(self.load_triplane_from_idx, range(len_tgt), ['tgt'] * len_tgt)

            len_latent = len(self.latent_path)
            with ThreadPoolExecutor() as executor:
                executor.map(self.load_triplane_from_idx, range(len_latent), ['latent'] * len_latent)

    def load_triplane_from_idx(self, idx, type: str):
        if type == 'src':
            triplane = load_triplane(self.src_tpath[idx])
            triplane = triplane.reshape(3, 32, self.resolution, self.resolution)
            triplane = torch.cat([triplane[0], triplane[1], triplane[2]], -1)
            self.src_triplanes[idx] = triplane
            return triplane
        elif type == 'tgt':
            triplane = load_triplane(self.tgt_tpath[idx])
            triplane = triplane.reshape(3, 32, self.resolution, self.resolution)
            triplane = torch.cat([triplane[0], triplane[1], triplane[2]], -1)
            self.tgt_triplanes[idx] = triplane
            return triplane
        elif type == 'latent':
            latent = torch.load(self.latent_path[idx])
            mean, logvar = torch.chunk(latent, 2, dim=0)
            self.latents[idx] = mean
            return latent
        else:
            raise ValueError(f"Unknown triplane type: {type}")

    def __getitem__(self, idx):
        src_id, tgt_id = self.pair_ids[idx]
        tgt_base_id = os.path.splitext(os.path.basename(self.tgt_tpath[tgt_id]))[0]
        if not self.preload_mm:
            src_t = self.load_triplane_from_idx(src_id, 'src')
            tgt_t = self.load_triplane_from_idx(tgt_id, 'tgt')
            latent = self.load_triplane_from_idx(tgt_id, 'latent')
            return src_t, tgt_t, latent, tgt_base_id
        else:
            return self.src_triplanes[src_id], self.tgt_triplanes[tgt_id], self.latents[tgt_id], tgt_base_id

    def __len__(self):
        return len(self.pair_ids)

    def dataloader(self):
        def custom_collate_fn(batch):
            src_t, tgt_t, latent, id = zip(*batch) # tuple
            return (torch.stack(src_t).to(self.device),
                    torch.stack(tgt_t).to(self.device),
                    torch.stack(latent).to(self.device), id)
        return DataLoader(self, batch_size = self.batch_size, shuffle = False, collate_fn = custom_collate_fn)

class TriplaneImagesDataset(Dataset):
    def __init__(self,
                 opt,
                 src_root,
                 src_data,
                 tgt_data,
                 all_ids,
                 device,
                 resolution,
                 local_rank,
                 world_size,
                 preload = True):
        self.src_root = src_root
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.all_ids = all_ids
        self.device = device
        self.batch_size = 1
        self.opt = opt
        self.resolution = resolution
        self.preload = preload
        self.all_ids = []
        self.src_photos = []
        self.ref_photos = []

        for id in all_ids:
            base_id = os.path.basename(id)

            # NOTE: the first img is front.
            src_photo_path = os.path.join(self.src_data, base_id, 'img_proc_fg_000000.png')
            src_img = load_image_rgb(src_photo_path)

            tgt_files = glob.glob(os.path.join(tgt_data, base_id + '_*'))
            for tgt_triplane_path in tgt_files:
                tgt_base_id = os.path.splitext(os.path.basename(tgt_triplane_path))[0]

                tgt_photo_path = os.path.join(self.tgt_data, tgt_base_id, 'img_proc_fg_000000.png')
                self.src_photos.append(src_img)
                self.ref_photos.append(load_image_rgb(tgt_photo_path))
                self.all_ids.append([base_id, tgt_base_id])
                break
            break

        self.all_ids = self.all_ids[local_rank:][::world_size]
        print(f'shard {local_rank}/{world_size} processing {len(self.all_ids)} avatars.')

        if(self.preload):
            self.train_loaders = []
            self.triplanes = []
            for src_id, tgt_id in self.all_ids:
                triplane, train_loader = self.load_from_disk(src_id, tgt_id)

                self.train_loaders.append(train_loader)
                self.triplanes.append(triplane)

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        src_id, tgt_id = self.all_ids[idx]

        if self.preload:
            return self.triplanes[idx], self.src_photos[idx], self.ref_photos[idx], self.train_loaders[idx], tgt_id

        triplane, train_loader = self.load_from_disk(src_id, tgt_id)

        return triplane, self.src_photos[idx], self.ref_photos[idx], train_loader, tgt_id

    def load_from_disk(self, src_id, tgt_id):
        train_dataset = NeRFDataset(self.opt, root_path=os.path.join(self.tgt_data, tgt_id),
                                            save_dir=self.opt.save_dir, device=self.device, type='train',
                                            triplane_resolution=self.opt.resolution0,
                                            triplane_channels=self.opt.triplane_channels,
                                            downscale=self.opt.downscale, num_train_frames=None)
        train_loader, _, _ = train_dataset.dataloader(face = True)
        triplane_path = os.path.join(self.src_root, src_id + '.npy')

        print("Loading triplane from {}".format(triplane_path))
        with open(triplane_path, 'rb') as f:
            triplane = np.load(f)
            triplane = torch.as_tensor(triplane, dtype=torch.float32)

        return triplane.unsqueeze(0), train_loader

    def dataloader(self):
        def collate_fn(batch):
            return batch[0]
        return DataLoader(self, batch_size = self.batch_size, shuffle=False, collate_fn=collate_fn)