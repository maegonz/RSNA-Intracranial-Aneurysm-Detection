import torch
import torch.nn as nn

from ublock import DoubleConv, EncodeSample, DecodeSample

class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encode_conv_1 = EncodeSample(in_channels, 64)
        self.encode_conv_2 = EncodeSample(64, 128)
        self.encode_conv_3 = EncodeSample(128, 256)
        self.encode_conv_4 = EncodeSample(256, 512)

        self.bottle_neck = DoubleConv(512,1024)

        self.decode_conv_1 = DecodeSample(1024, 512)
        self.decode_conv_2 = DecodeSample(512, 256)
        self.decode_conv_3 = DecodeSample(256, 128)
        self.decode_conv_4 = DecodeSample(128, 64)

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