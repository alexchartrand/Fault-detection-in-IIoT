# FCNNs
#Wang Z, Yan W, Oates T (2017b) Time series classification from scratch with deep neural networks:
#A strong baseline. In: International Joint Conference on Neural Networks, pp 1578–1585

import torch.nn as nn
import torch.optim as optim
import Utility

class FCNNs(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(FCNNs, self).__init__()

        #self.activation = nn.Softmax(dim=1) # This is include in the loss
        self.net = nn.Sequential(Utility.ConvBlock(in_feature, 128, 8),
                                Utility.ConvBlock(128, 256, 5),
                                Utility.ConvBlock(256, 128, 3),
                                Utility.GAP(),
                                nn.Linear(128, out_feature))

    def forward(self, x):
        return self.net(x)

    def getOptimizer(self):
        return optim.Adam(self.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-8)
