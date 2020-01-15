# FCNNs
#Wang Z, Yan W, Oates T (2017b) Time series classification from scratch with deep neural networks:
#A strong baseline. In: International Joint Conference on Neural Networks, pp 1578–1585

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FCNNs(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(FCNNs, self).__init__()

        #self.activation = nn.Softmax(dim=1) # This is include in the loss
        self.net = nn.Sequential(FCNNsBlock(in_feature, 128, 8),
                                FCNNsBlock(128, 256, 5),
                                FCNNsBlock(256, 128, 3),
                                GAP(),
                                nn.Linear(128, out_feature))

    def forward(self, x):
        return self.net(x)

    def getOptimizer(self):
        return optim.Adam(self.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-8)


class FCNNsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(FCNNsBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=2)
        return x