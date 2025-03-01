import os
import torch
import numpy as np

from torch import nn

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


class CustomMSELoss(nn.Module):
    def __init__(self, threshold=0.01, scale_factor=0.1):
        super(CustomMSELoss, self).__init__()
        self.threshold = threshold
        self.scale_factor = scale_factor

    def forward(self, prediction, target):
        abs_error = torch.abs(prediction - target)

        mask = abs_error < self.threshold
        scaled_error = torch.where(mask, abs_error * self.scale_factor, abs_error)

        loss = torch.mean(scaled_error ** 2)
        return loss