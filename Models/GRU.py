import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
        """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dp_keep_prob):
        super().__init__()
        self.dropout = nn.Dropout(p=1 - dp_keep_prob)
        self.Wr = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.Wz = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = torch.nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        inputs = self.dropout(inputs)
        r = torch.sigmoid(self.Wr(torch.cat([inputs, hidden], 1)))
        z = torch.sigmoid(self.Wz(torch.cat([inputs, hidden], 1)))
        h_hat = torch.tanh(self.Wh(torch.cat([inputs, r * hidden], 1)))
        out = (1 - z) * hidden + z * h_hat
        return out

# Code taken from IFT6135 class at University of Montreal

class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        # TODO ========================
        # Parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        # layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        first_gru_block = GRUBlock(emb_size, hidden_size, dp_keep_prob)
        additional_gru_blocks = clones(GRUBlock(hidden_size, hidden_size, dp_keep_prob), num_layers - 1)
        self.gru_blocks = nn.ModuleList([first_gru_block]).extend(additional_gru_blocks)
        self.output_dropout = nn.Dropout(p=1 - dp_keep_prob)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        self.init_weights_uniform()

    def init_weights_uniform(self):
        # TODO ========================
        torch.nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)

        k = np.sqrt(1. / self.hidden_size)
        for layer in self.gru_blocks:
            torch.nn.init.uniform_(layer.Wr.weight, a=-k, b=k)
            torch.nn.init.uniform_(layer.Wr.bias, a=-k, b=k)
            torch.nn.init.uniform_(layer.Wz.weight, a=-k, b=k)
            torch.nn.init.uniform_(layer.Wz.bias, a=-k, b=k)
            torch.nn.init.uniform_(layer.Wh.weight, a=-k, b=k)
            torch.nn.init.uniform_(layer.Wh.bias, a=-k, b=k)

        torch.nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        torch.nn.init.constant_(self.output_layer.bias, 0.)

    def init_hidden(self):
        # TODO ========================
        return torch.Tensor(self.num_layers, self.batch_size, self.hidden_size).fill_(0.)

    def forward(self, inputs, hidden):
        # TODO ========================
        logits = []
        for xbt in inputs:
            out = self.embedding(xbt)
            next_hidden = []
            for i, layer in enumerate(self.gru_blocks):
                out = layer(out, hidden[i])
                next_hidden.append(out)
            out = self.output_dropout(out)
            out = self.output_layer(out)
            logits.append(out)
            hidden = next_hidden

        logits = torch.stack(logits)
        hidden = torch.stack(hidden)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================

        samples = []
        x_hat = input
        for _ in generated_seq_len:
            out = self.embedding(x_hat)
            next_hidden = []
            for i, layer in enumerate(self.rnn_blocks):
                out = layer(out, hidden[i])
                next_hidden.append(out)
            out = self.output_dropout(out)
            out = self.output_layer(out)
            out = torch.softmax(out)
            sample = torch.distributions.categorical.Categorical(out).sample()
            samples.append(sample)
            hidden = next_hidden

        samples = torch.stack(samples)
        return samples

# Code taken from: https://blog.floydhub.com/gru-with-pytorch/

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden