# https://arxiv.org/pdf/1801.04503.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import math
from . import Utility

# Inspired by: https://arxiv.org/pdf/1801.04503.pdf
class CNNGRU_PAR(nn.Module):
    HIDDEN_SIZE = 512
    DROPOUT = 0.2
    DOWN_SAMPLE = 10
    def __init__(self, in_feature, out_feature):
        super(CNNGRU_PAR, self).__init__()
        self.res1 = ResBlock(in_feature,60, 5)
        self.res2 = ResBlock(60, 120, 3)
        self.res3 = ResBlock(120, 240, 3)
        self.res4 = ResBlock(240, 240, 3)
        self.gap = Utility.GAP()

        self.downSample = nn.AvgPool1d(self.DOWN_SAMPLE, self.DOWN_SAMPLE)
        self.lstm = nn.LSTM(input_size=in_feature, hidden_size=self.HIDDEN_SIZE, num_layers=3, bidirectional=True, batch_first=True, dropout=self.DROPOUT)

        concatSize = self.HIDDEN_SIZE*2+240
        self.lin1 = nn.Linear(concatSize, self.HIDDEN_SIZE)
        self.bn2 = nn.BatchNorm1d(self.HIDDEN_SIZE)
        self.lin2 = nn.Linear(self.HIDDEN_SIZE, out_feature)
        self._init_weight()

    def _init_weight(self):

        # GRU initialization [-1/32, 1/32]
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.ones_(param)
            elif 'weight' in name:
                nn.init.uniform_(param, -1/32, 1/32)
        # Linear
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        p1 = self.res1(x)
        p1 = self.res2(p1)
        p1 = self.res3(p1)
        p1 = self.res4(p1)
        p1 = self.gap(p1)

        p2 = self.downSample(x)
        p2 = p2.permute(0,2,1)
        p2, _ = self.lstm(p2)
        p2 = p2[:,-1,:]

        concat = torch.cat((p1, p2), 1)
        lin_out = self.lin1(concat)
        lin_out = self.bn2(lin_out)
        lin_out = nn.ReLU()(lin_out)
        y = self.lin2(lin_out)
        return y

    def getOptimizer(self):
        return optim.SGD(self.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-5)

class ResBlock(nn.Module):

    def __init__(self, in_feature, out_feature, kernel):
        super(ResBlock, self).__init__()
        self.need_identity = False
        if in_feature != out_feature:
            self.need_identity = True
            self.convIdentity = Utility.Conv1DSame(in_feature, out_feature, 1, groups=in_feature, bias=False)

        self.identityBN = nn.BatchNorm1d(out_feature)
        self.bn1 = nn.BatchNorm1d(out_feature)
        self.conv1 = Utility.Conv1DSame(in_feature, out_feature, kernel, groups=in_feature)
        self.conv2 = Utility.Conv1DSame(out_feature, out_feature, kernel, groups=out_feature)
        self.conv3 = Utility.Conv1DSame(out_feature, out_feature, kernel, groups=out_feature)
        self._init_weigth()

    def _init_weigth(self):
        if self.need_identity:
            nn.init.ones_(self.convIdentity.conv.weight)

    def forward(self, x):
        identity = x
        if self.need_identity:
            identity = self.convIdentity(x)

        identity = self.identityBN(identity)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn1(out)

        res = out + identity
        return nn.ReLU()(res)
