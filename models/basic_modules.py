import torch.nn.functional as F
import torch.nn as nn
import torch

class BlurModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1,
                 padding=1):
        super(BlurModule, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, bias=False)
        nn.init.constant_(self.conv.weight, 1/(kernel_size[0]*kernel_size[1]))

    def forward(self, input):
        return self.conv(input)


class GammaModule(nn.Module):
    def __init__(self, gamma=0.7):
        super(GammaModule, self).__init__()
        self.gamma = gamma

    def forward(self, input):
        return input.pow(self.gamma)