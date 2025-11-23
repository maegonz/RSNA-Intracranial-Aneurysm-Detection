import torch
import torch.nn as nn

class DoubleConv3d(nn.Module):
    """
    Basic block of a UNet architecture, which applies two consecutive 3D convolutional
    layers, each followed by a ReLU activation.

    Params
    -------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for both convolution layers.

    Shape
    -------
    Input:
        (N, in_channels, D, H, W)
    Output:
        (N, out_channels, D, H, W)

    Returns
    -------
    torch.Tensor
        The output feature map after two convolutions and activations.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.operation(x)
        return y

class Encoder3d(nn.Module):
    """
    3D encoder block consisting of a DoubleConv3d followed by 3D max pooling.

    The convolutional block extracts features, while the max pooling down-samples
    the spatial dimensions by a factor of 2. Both the convolution output and the
    pooled output are returned to support skip connections.

    Paras
    -------
    in_channels : int
        Number of channels for the input tensor.
    out_channels : int
        Number of output channels after the DoubleConv3d block.

    Shape
    -------
    Input:
        (N, in_channels, D, H, W)
    conv_output:
        (N, out_channels, D, H, W)
    pooled_output:
        (N, out_channels, D/2, H/2, W/2)

    Returns
    -------
    c, p : tuple of torch.Tensor
        - conv_output : Features from DoubleConv3d.
        - pooled_output : Downsampled features.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv3d(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p


class Decoder3d(nn.Module):
    """
    3D decoder block consisting of a transposed convolution for upsampling,
    followed by a DoubleConv3d for feature refinement.

    The upsampled tensor is concatenated with a skip connection from the encoder,
    fusioning feature across scales.

    Params
    -------
    in_channels : int
        Number of channels for the concatenated input (upsampled + skip connection).
    out_channels : int
        Number of output channels after the DoubleConv3d block.

    Shape
    -------
    t1 :
        (N, in_channels/2, D/2, H/2, W/2)
        Deeper features from the decoder path.
    t2 :
        (N, in_channels/2, D, H, W)
        Skip connection from the encoder.
    Output :
        (N, out_channels, D, H, W)

    Returns
    -------
    torch.Tensor
        The refined decoder output after upsampling, concatenation, and DoubleConv3d.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, t1, t2):
        t1 = self.deconv(t1)
        
        x = torch.cat([t1, t2], 1)
        y = self.conv(x)
        return y