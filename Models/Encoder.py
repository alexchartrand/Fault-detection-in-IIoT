import torch
import torch.nn as nn
import torch.optim as optim
import math
from . import Utility

class Encoder(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(Encoder, self).__init__()

        # self.activation = nn.Softmax(dim=1) # This is include in the loss

        self.block1 = ConvBlock(in_feature, 128, 5)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = ConvBlock(128, 256, 11)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = ConvBlock(256, 512, 21)
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(256, 256)

        self.instanceNorm = nn.InstanceNorm1d(1250, affine=True)
        self.linearEnd = nn.Linear(256*1250, out_feature)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = x.permute(0, 2, 1)
        attentionData, attentionSoftmax = torch.split(x, int(x.shape[2]/2), dim=2)
        attentionSoftmax  = self.softmax(attentionSoftmax)
        # att = (a * h).sum(2)
        multiply_layer = attentionSoftmax * attentionData
        dense_layer = self.linear(multiply_layer)
        dense_layer = nn.Sigmoid()(dense_layer)
        dense_layer = self.instanceNorm(dense_layer)

        flatten_layer = torch.flatten(dense_layer, start_dim=1)
        return self.linearEnd(flatten_layer)

    def getOptimizer(self):
        return optim.Adam(self.parameters(), lr=0.00001)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = Utility.Conv1DSame(in_channels, out_channels, kernel_size)
        self.instanceNorm = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = nn.PReLU(out_channels)
        self.dropout = nn.Dropout(0.35)


    def forward(self, x):
        x = self.conv(x)
        x = self.instanceNorm(x)
        x = self.activation(x)
        out = self.dropout(x)
        return out