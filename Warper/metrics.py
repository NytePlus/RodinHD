import os
import torch
import numpy as np

from torch import nn
from id_model import Backbone

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class ArcfaceLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.id_model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.id_model.load_state_dict(torch.load(current_dir + "/weights/model_ir_se50.pth", map_location=device))
        self.id_model.eval()

    def forward(self, output, target):
        # print(output.shape, target.shape)
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1).reshape(-1, 3, 128, 128)
            target = target.permute(0, 2, 1).reshape(-1, 3, 128, 128)
        else:

            output = output.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)
        output = (output + 1.) / 2.
        target = (target + 1.) / 2.
        output = output.clamp(0, 1)
        target = target.clamp(0, 1)
        output = torch.nn.functional.interpolate(output, size=(112, 112), mode='bilinear', align_corners=False)
        target = torch.nn.functional.interpolate(target, size=(112, 112), mode='bilinear', align_corners=False)
        emb_output = self.id_model(output)
        emb_target = self.id_model(target)
        loss = torch.einsum('ij,ij->i', emb_output, emb_target)
        return loss