""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_shift import TemporalShift


class ShiftConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment, kernel_size, padding, stride, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.shiftconv = TemporalShift(self.conv, n_segment=n_segment, n_div=3)
        self.n_segment = n_segment

    def forward(self, x):
        """x = [bt, c, h ,w]"""
        return self.shiftconv(x)


class ShiftResConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment, kernel_size, padding, stride, dilation ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.shiftconv = TemporalShift(self.conv, n_segment=n_segment, n_div=3)

        self.resconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

        self.n_segment = n_segment

    def forward(self, x):
        """x = [bt, c, h ,w]"""
        _x = self.shiftconv(x)
        return _x + self.resconv(x)


class ShiftConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment ):
        super().__init__()
        self.conv = ShiftConv(in_channels, out_channels, n_segment, kernel_size=3,padding=1,stride=1,dilation=1)

    def forward(self, x):
        """x = [bt, c, h ,w]"""
        return self.conv(x)

class ShiftConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment ):
        super().__init__()
        self.conv = ShiftConv(in_channels, out_channels, n_segment, kernel_size=1,padding=0,stride=1,dilation=1)

    def forward(self, x):
        """x = [bt, c, h ,w]"""
        return self.conv(x)


class ShiftResConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment ):
        super().__init__()
        self.conv = ShiftResConv(in_channels, out_channels, n_segment, kernel_size=3,padding=1,stride=1,dilation=1)

    def forward(self, x):
        """x = [bt, c, h ,w]"""
        return self.conv(x)

class ShiftResConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment ):
        super().__init__()
        self.conv = ShiftResConv(in_channels, out_channels, n_segment, kernel_size=1,padding=0,stride=1,dilation=1)

    def forward(self, x):
        """x = [bt, c, h ,w]"""
        return self.conv(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_segment=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU()
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = ShiftConv3x3(planes, planes, n_segment=n_segment)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # print(n_segment)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        if use_bn:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2dShift(nn.Module):
    """input = [bt,c,h,w]"""
    def __init__(self, in_channels, out_channels, n_segment):
        super().__init__()
        self.double_conv = nn.Sequential(
            # ShiftResConv3x3(in_channels, out_channels, n_segment),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ShiftResConv3x3(out_channels, out_channels, n_segment),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down3d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down2dShift(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, n_segment):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #考虑变成3d
            DoubleConv2dShift(in_channels, out_channels, n_segment)
        )

    def forward(self, x):
        """[bt,c,h,w]"""
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_bn=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size(2) - x1.size(2)])
        diffX = torch.tensor([x2.size(3) - x1.size(3)])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCTHW
        diffY = torch.tensor([x2.size(3) - x1.size(3)])
        diffX = torch.tensor([x2.size(4) - x1.size(4)])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConv3d(nn.Module):
    """
    input x [b,cin,t,h,w]
    output [b,cout,t,h,w]
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv2dShift(nn.Module):
    """
    input x [bt,cin,h,w]
    output [bt,cout,h,w]
    """
    def __init__(self, in_channels, out_channels, n_segment):
        super(OutConv2dShift, self).__init__()
        self.conv = ShiftResConv1x1(in_channels, out_channels,n_segment)

    def forward(self, x):
        return self.conv(x)