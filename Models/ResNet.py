# Resnet
#Wang Z, Yan W, Oates T (2017b) Time series classification from scratch with deep neural networks:
#A strong baseline. In: International Joint Conference on Neural Networks, pp 1578–1585

import torch
import torch.nn as nn
import torch.optim as optim
import math
import Utility

class ResNet(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(ResNet, self).__init__()

        # self.activation = nn.Softmax(dim=1) # This is include in the loss
        self.net = nn.Sequential(nn.BatchNorm1d(in_feature),
                                 ResBlock(in_feature, 64),
                                 ResBlock(64, 128),
                                 ResBlock(128, 128),
                                 Utility.GAP(),
                                 nn.Linear(128, out_feature))

    def forward(self, x):
        return self.net(x)

    def getOptimizer(self):
        return optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.need_reshape_input = False
        if in_channels != out_channels:
            self.need_reshape_input = True
            self.identityConv = nn.Conv1d(in_channels, out_channels, 1)

        self.identityBN = nn.BatchNorm1d(out_channels)
        self.block1 = Utility.ConvBlock(in_channels, out_channels, 8)
        self.block2 = Utility.ConvBlock(out_channels, out_channels, 5)
        self.block3 = Utility.ConvBlock(out_channels, out_channels, 3, activation=None)

    def forward(self, x):
        identity = x
        if self.need_reshape_input:
            identity = self.identityConv(identity)
        identity = self.identityBN(identity)
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        y = h3 + identity
        return nn.ReLU()(y)