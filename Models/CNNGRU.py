# https://arxiv.org/pdf/1712.07108.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import math
from . import Utility

# Inspired by: https://arxiv.org/pdf/1712.07108.pdf
class CNNGRU(nn.Module):
    HIDDEN_SIZE = 512
    DROPOUT = 0.3
    def __init__(self, in_feature, out_feature):
        super(CNNGRU, self).__init__()
        self.bnx = nn.BatchNorm1d(in_feature)
        self.convIdentity = Utility.Conv1DSame(in_feature, 64, 1, bias=False)
        self.res1 = ResBlock(64,64, 5)
        self.res2 = ResBlock(64, 128, 3)
        self.res3 = ResBlock(128, 256, 3)
        self.res4 = ResBlock(256, 512, 3)
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.HIDDEN_SIZE, num_layers=3, bidirectional=True, batch_first=True, dropout=self.DROPOUT)
        self.bn1 = nn.BatchNorm1d(self.HIDDEN_SIZE*2)
        self.lin1 = nn.Linear(self.HIDDEN_SIZE*2, self.HIDDEN_SIZE)
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

        nn.init.ones_(self.convIdentity.conv.weight)
        # Linear
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        out = self.bnx(x)
        out = self.convIdentity(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        gru_in = out.permute(0,2,1)
        gru_out, _ = self.lstm(gru_in)
        gru_out = gru_out[:,-1,:]
        gru_out = self.bn1(gru_out)

        lin_out = self.lin1(gru_out)
        lin_out = self.bn2(lin_out)
        lin_out = nn.ReLU()(lin_out)
        y = self.lin2(lin_out)
        return y

    def getOptimizer(self):
        return optim.SGD(self.parameters(), lr=0.1, momentum=0.95, nesterov=True, weight_decay=1e-5)

class DownSampleConv(nn.Module):

    def __init__(self, in_feature, out_feature, kernel):
        super(DownSampleConv, self).__init__()
        self.conv = nn.Conv1d(in_feature, out_feature, kernel_size=kernel, stride=2, padding=kernel//2, groups=in_feature)
        self.bn = nn.BatchNorm1d(out_feature)
        self._init_weigth()

    def _init_weigth(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU()(x)

class ResBlock(nn.Module):

    def __init__(self, in_feature, out_feature, kernel):
        super(ResBlock, self).__init__()
        self.need_identity = False
        stride = int(out_feature/in_feature)
        if in_feature != out_feature:
            self.need_identity = True
            self.convIdentity = Utility.Conv1DSame(in_feature, out_feature, 1, stride=stride, groups=in_feature, bias=False)

        self.identityBN = nn.BatchNorm1d(out_feature)
        self.bn1 = nn.BatchNorm1d(out_feature)
        self.conv1 = Utility.Conv1DSame(in_feature, out_feature, kernel, stride=stride, groups=in_feature)
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
