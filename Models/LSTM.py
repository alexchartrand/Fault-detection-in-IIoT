import torch
import torch.nn as nn
import torch.optim as optim
import math
from . import Utility

class LSTM(nn.Module):
    HIDDEN_SIZE = 512
    DROPOUT = 0.3
    def __init__(self, in_feature, out_feature):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_feature, self.HIDDEN_SIZE, num_layers=2, batch_first=True, bidirectional=True, dropout=self.DROPOUT)
        self.dropout1 = nn.Dropout(self.DROPOUT)
        self.lin1 = nn.Linear(self.HIDDEN_SIZE*2, self.HIDDEN_SIZE)
        self.dropout2= nn.Dropout(self.DROPOUT)
        self.lin2 = nn.Linear(self.HIDDEN_SIZE, out_feature)
        self._init_weigth()

    def _init_weigth(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.uniform_(param, -1/32, 1/32)
        # Linear
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        rnn_in = x.permute(0, 2, 1)
        rnn_out, _ = self.lstm(rnn_in)
        rnn_out = rnn_out[:, -1, :]
        rnn_out = self.dropout1(rnn_out)

        lin_out = self.lin1(rnn_out)
        lin_out = nn.ReLU()(lin_out)
        lin_out = self.dropout2(lin_out)

        return self.lin2(lin_out)

    def getOptimizer(self):
        return optim.SGD(self.parameters(), lr=0.1, momentum=0.95, nesterov=True, weight_decay=1e-5)