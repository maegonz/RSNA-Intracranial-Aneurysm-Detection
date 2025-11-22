import torch
import torch.nn as nn
from ublock3d import *

class AnNet(nn.Module):
    """
    3D U-Net based model with dual outputs for classification and segmentation.

    This architecture consists of:
    - **Encoder**: A four-level 3D U-Net encoder that progressively downsamples 
    the input volume and extracts hierarchical feature representations.
    - **Bottleneck**: A double-convolution block at the lowest resolution.
    - **Classification head**: A global average pooling layer followed by a 
    fully connected layer that produces a single scalar prediction for binary
    classification whether there is aneurysm present or not.
    - **Segmentation decoder**: A four-stage upsampling path that mirrors the 
    encoder and produces a voxel-wise segmentation map via a final 1x1x1 
    convolution in order to localize aneurysm.

    Params
    -------
    in_channels : int
        Number of input channels in the 3D volume.

    Notes
    -------
    This module outputs:
    1. A classification score (shape: `[batch, 1]`)
    2. A segmentation mask (shape: `[batch, 1, D, H, W]`)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # ----- Encoder -----
        self.encode_1 = Encoder3d(in_channels, 32)
        self.encode_2 = Encoder3d(32, 64)
        self.encode_3 = Encoder3d(64, 128)
        self.encode_4 = Encoder3d(128, 256)
        self.bottle_neck = DoubleConv3d(256, 512)

        # ----- Classification head -----
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)

        # ----- Segmentation decoder -----
        self.decode_1 = Decoder3d(512, 256)
        self.decode_2 = Decoder3d(256, 128)
        self.decode_3 = Decoder3d(128, 64)
        self.decode_4 = Decoder3d(64, 32)
        self.seg_out = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Encoder
        down1, p1 = self.encode_1(x)
        down2, p2 = self.encode_2(p1)
        down3, p3 = self.encode_3(p2)
        down4, p4 = self.encode_4(p3)
        b = self.bottle_neck(p4)

        # Classification
        pooled = self.global_pool(b).flatten(1)
        clf_logits = self.fc(pooled)

        # Segmentation        
        up1 = self.decode_1(b, down4)
        up2 = self.decode_2(up1, down3)
        up3 = self.decode_3(up2, down2)
        up4 = self.decode_4(up3, down1)
        seg_mask = self.seg_out(up4)
        seg_mask = torch.sigmoid(seg_mask)

        return clf_logits, seg_mask

