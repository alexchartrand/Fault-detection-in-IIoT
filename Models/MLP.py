# FCNNs
#Wang Z, Yan W, Oates T (2017b) Time series classification from scratch with deep neural networks:
#A strong baseline. In: International Joint Conference on Neural Networks, pp 1578–1585

import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(MLP, self).__init__()

        #self.activation = nn.Softmax(dim=1) # This is include in the loss
        self.net = nn.Sequential(MLPBlock(in_feature, 500, 0.1),
                                MLPBlock(500, 500, 0.2),
                                MLPBlock(500, out_feature, 0.2),
                                 nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x)

    def getOptimizer(self):
        return optim.Adadelta(self.parameters(), lr=0.1, eps=1e-8, rho=0.95)

class MLPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super(MLPBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x