from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from mpi4py import MPI
import sys
sys.path.append("../")
import argparse
from pretrained_diffusion import dist_util
from modules.emonet import EmoNet


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
        # path = os.path.join(self.root, path_base, "0000.png")
       
        pil_image2 = Image.open(path).resize((self.resolution, self.resolution), Image.LANCZOS)
        image = self.get_input_image_tensor(pil_image2)
        
        data_dict = {"path": path_base.split("/")[-1]}
        return image, data_dict
    
    def get_input_image_tensor(self, image):
        im_data = np.array(image.convert("RGBA")).astype(np.float32)
        bg = np.array([1,1,1]).astype(np.float32)
        norm_data = im_data / 255.0
        alpha = norm_data[:, :, 3:4]
        arr = norm_data[:,:,:3] * alpha + bg * (1 - alpha)
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) * 2.0 - 1.0
        return image_tensor


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
parser.add_argument("--vae_dir", type=str, default="../weights/emonet_8.pth")
args = parser.parse_args()

model = EmoNet().to(dist_util.dev())
model.load_state_dict(torch.load(args.vae_dir))

batch_size = 4
dataset = ImageDataset(args.root, txt_file=args.txt_file, start_idx=args.start_idx, end_idx=args.end_idx, resolution=256)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,)
os.makedirs(os.path.join(args.output_dir, "ms_latent"), exist_ok=True)

for i, (images, data_dicts) in enumerate(tqdm(dataloader)):  
    images = images.to(dist_util.dev())  
  
    with torch.no_grad():  
        feature, feat_dict = model.encode_feat(images)

        feat_list = [f for f in feat_dict.values()]

        feats_fp16, feats_fp32 = separate_features(feat_list)

    for j, latent_sample_fp32 in enumerate(feature):
        output_path_fp32 = os.path.join(args.output_dir, data_dicts["path"][j]+".pt")
        torch.save(latent_sample_fp32, output_path_fp32)

    for j, latent_sample_fp32 in enumerate(feats_fp32):
        ms_output_path_fp32 = os.path.join(args.output_dir, 'ms_latent', data_dicts["path"][j]+".pt")
        torch.save(latent_sample_fp32, ms_output_path_fp32)