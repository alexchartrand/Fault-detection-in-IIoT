import torch
import torch.nn as nn
import torch.optim as optim
import  torch.functional as F

class LSTM(nn.Module):
    HIDDEN_SIZE = 512
    DROPOUT = 0.1
    DOWN_SAMPLE = 10
    def __init__(self, in_feature, out_feature):
        super(LSTM, self).__init__()
        self.downSample = nn.AvgPool1d(self.DOWN_SAMPLE, self.DOWN_SAMPLE)
        self.lstm = nn.LSTM(in_feature, self.HIDDEN_SIZE, num_layers=3, batch_first=True, bidirectional=True, dropout=self.DROPOUT)
        self.dropout2 = nn.Dropout(self.DROPOUT)
        self.lin1 = nn.Linear(self.HIDDEN_SIZE*2, self.HIDDEN_SIZE)
        self.dropout3 = nn.Dropout(self.DROPOUT)
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
        x = self.downSample(x)
        rnn_in = x.permute(0, 2, 1)
        rnn_out, _ = self.lstm(rnn_in)
        rnn_out = rnn_out[:, -1, :]
        rnn_out = self.dropout2(rnn_out)

        lin_out = self.lin1(rnn_out)
        lin_out = nn.ReLU()(lin_out)
        lin_out = self.dropout3(lin_out)

        return self.lin2(lin_out)

    def getOptimizer(self):
        return optim.SGD(self.parameters(), lr=0.1, momentum=0.9, nesterov=True)

class DownSampleConv(nn.Module):

    def __init__(self, in_feature, out_feature, kernel):
        super(DownSampleConv, self).__init__()
        self.conv = nn.Conv1d(in_feature, out_feature, kernel_size=kernel, stride=2, padding=kernel//2, groups=in_feature)
        self._init_weigth()

    def _init_weigth(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        return nn.ReLU()(x)