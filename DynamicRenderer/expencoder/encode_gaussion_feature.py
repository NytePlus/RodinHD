from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from mpi4py import MPI
import sys
sys.path.append("../../")
import argparse
from pretrained_diffusion import dist_util
from DynamicRenderer.expencoder.exp import ExpEncoder


class ImageDataset(Dataset):
    def __init__(
        self,
        root,
        txt_file='',
        resolution=128,
        start_idx=0,
        end_idx=100,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    ):
        super().__init__()
        self.root = root
        self.txt_file = txt_file
        self.resolution = resolution
        self.local_images = self.get_all_file(start_idx, end_idx)[shard:][::num_shards]
        print("Total images: ", len(self.local_images))
    
    def get_all_file(self, start, end):
        with open(self.txt_file) as f:
            all_files = f.read().splitlines()[start:end]

        return all_files

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path_base = self.local_images[idx]
        path = os.path.join(self.root, path_base, "img_proc_fg_000000.png")
       
        pil_image2 = Image.open(path).resize((self.resolution, self.resolution), Image.LANCZOS)
        image = self.get_input_image_tensor(pil_image2)
        
        data_dict = {"path": path_base.split("/")[-1]}
        return image, data_dict

    def get_input_image_tensor(self, img) -> torch.Tensor:
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
        return x

def separate_features(input_list):  
    separated_features_fp16 = []  
    separated_features_fp32 = []  
    batch_size = input_list[0].size(0)

    for i in range(batch_size):
        feat_i_fp16 = []
        feat_i_fp32 = []
        for tensor in input_list:
            feat_i_fp16.append(tensor[i].cpu().to(torch.float16))
            feat_i_fp32.append(tensor[i].cpu().to(torch.float32))
        separated_features_fp16.append(feat_i_fp16)
        separated_features_fp32.append(feat_i_fp32)
  
    return separated_features_fp16, separated_features_fp32


dist_util.setup_dist()

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str)
parser.add_argument("--txt_file", type=str)
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

model = ExpEncoder(dist_util.dev())

batch_size = 4
dataset = ImageDataset(args.root, txt_file=args.txt_file, start_idx=args.start_idx, end_idx=args.end_idx)  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,)  
os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)

for i, (images, data_dicts) in enumerate(tqdm(dataloader)):  
    images = images.to(dist_util.dev())  
  
    with torch.no_grad():  
        feats = model.encode_feat(images, [1, 128, 128])

    for j, feat in enumerate(feats):
        output_path = os.path.join(args.output_dir, "latent", data_dicts["path"][j]+".pt")
        torch.save(feat, output_path)

