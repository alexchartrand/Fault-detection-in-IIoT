import torch
import torch.nn as nn
import math

class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=2)
        return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv = Conv1DSame(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv1DSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(Conv1DSame, self).__init__()
        p = (kernel_size-1)/2
        self.padding = nn.ConstantPad1d((math.floor(p), math.ceil(p)), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups)

    def forward(self, x):
        x =self.padding(x)
        out = self.conv(x)
        return  out
