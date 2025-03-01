import yaml
import torch
from torch import nn

from Warper.LivePortrait.utils.helper import load_model
from Warper.LivePortrait.modules.util import kp2gaussian
from Warper.LivePortrait.config.inference_config import InferenceConfig

class ExpEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()

        model_config = yaml.load(open(InferenceConfig.models_config, 'r'), Loader=yaml.SafeLoader)
        self.motion_extractor = load_model(InferenceConfig.checkpoint_M, model_config, device, 'motion_extractor')

    def encode_feat(self, x, spatial_size): # (d, h, w)
        bs, c, h, w = x.shape
        with torch.no_grad():
            kp_info = self.motion_extractor(x)
        kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)
        gaussion_feat = kp2gaussian(kp_info['kp'], spatial_size=spatial_size, kp_variance=1) # (bs, num_kp, d, h, w)

        return kp_info['kp'], gaussion_feat.reshape(-1, 21, 128, 128)[:, :32]