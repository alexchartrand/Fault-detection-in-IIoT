# FCNNs
#Wang Z, Yan W, Oates T (2017b) Time series classification from scratch with deep neural networks:
#A strong baseline. In: International Joint Conference on Neural Networks, pp 1578–1585

import torch.nn as nn
import torch.optim as optim
from . import Utility

class FCNNs(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(FCNNs, self).__init__()

        #self.activation = nn.Softmax(dim=1) # This is include in the loss
        self.conv1 = Utility.ConvBlock(in_feature, 128, 8)
        self.conv2 = Utility.ConvBlock(128, 256, 5)
        self.conv3 = Utility.ConvBlock(256, 128, 3)
        self.gap = Utility.GAP()
        self.lin = nn.Linear(128, out_feature)

    def forward(self, x):
        h=self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.gap(h)
        h = self.lin(h)
        return h

    def getOptimizer(self):
        return optim.Adam(self.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-7)
