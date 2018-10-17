import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Bi-directional RNN model with GRU
"""
class Bid_RNN(nn.Module):
    def __init__(self, in_size, hid_size, class_num=2):
        super(Bid_RNN ,self).__init__()
        self.rnn = nn.GRU(input_size=in_size, hidden_size=hid_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hid_size*2, class_num)
        self.hid_size = hid_size

    ## x: length X 2
    def forward(self, x):
        h0 = Variable(torch.zeros(2, x.size(0), self.hid_size).cuda())
        #c0 = Variable(torch.zeros(2, x.size(0), self.hid_size).cuda())
        out, _ = self.rnn(x, h0)
        out = out.squeeze()
        out = self.linear(out)
        return out


