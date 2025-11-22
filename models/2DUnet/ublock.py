import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.operation(x)
        return y
    

class EncodeSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(EncodeSample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, )

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p
    

class DecodeSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DecodeSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, t1, t2):
        t1 = self.deconv(t1)
        
        x = torch.cat([t1, t2], 1)
        y = self.conv(x)
        return y