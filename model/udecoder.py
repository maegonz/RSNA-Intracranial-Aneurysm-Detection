import torch
import torch.nn as nn

from uencoder import DoubleConv

class DecodeSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2D(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, t1, t2):
        t1 = self.deconv(t1)
        
        x = torch.cat([t1, t2], 1)
        y = self.conv(x)
        return y