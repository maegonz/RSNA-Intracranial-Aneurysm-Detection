import torch
import torch.nn as nn

from models.uNet.ublock2d import DoubleConv2d, Encoder2d, Decoder2d

class UNet2d(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encode_conv_1 = Encoder2d(in_channels, 64)
        self.encode_conv_2 = Encoder2d(64, 128)
        self.encode_conv_3 = Encoder2d(128, 256)
        self.encode_conv_4 = Encoder2d(256, 512)

        self.bottle_neck = DoubleConv2d(512,1024)

        self.decode_conv_1 = Decoder2d(1024, 512)
        self.decode_conv_2 = Decoder2d(512, 256)
        self.decode_conv_3 = Decoder2d(256, 128)
        self.decode_conv_4 = Decoder2d(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down1, p1 = self.encode_conv_1(x)
        down2, p2 = self.encode_conv_2(p1)
        down3, p3 = self.encode_conv_3(p2)
        down4, p4 = self.encode_conv_4(p3)

        b = self.bottle_neck(p4)

        up1 = self.decode_conv_1(b, down4)
        up2 = self.decode_conv_2(up1, down3)
        up3 = self.decode_conv_3(up2, down2)
        up4 = self.decode_conv_4(up3, down1)

        y = self.out(up4)
        return y