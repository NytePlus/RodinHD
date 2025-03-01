import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels = [32, 64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        blks = [
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for i in range(len(hidden_channels) - 1):
            blks.extend([
                nn.Conv2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.model = nn.Sequential(*blks)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[-1], out_channels=1, kernel_size=(4, 12), stride=1, padding=0),
        )

    def forward(self, x):
        bs = x.shape[0]
        y = self.model(x)
        return self.output(y).reshape(bs, -1)