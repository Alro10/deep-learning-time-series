'''This is the model for taxi drivers dataset'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.bi = 1
        self.lstm = nn.LSTM(config.get('features'),self.hidden_size,1,dropout=0.1,bidirectional=self.bi-1,batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size,self.hidden_size // 4,1,dropout=0.1,bidirectional=self.bi-1,batch_first=True)
        self.dense = nn.Linear(self.hidden_size // 4, config.get('forecast_horizon'))
        self.loss_fn = nn.MSELoss()

    def forward(self, x, batch_size=100):
        hidden = self.init_hidden(batch_size))
        output, _ = self.lstm(x, hidden)
        output = F.dropout(output, p=0.5, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.dense(state[0].squeeze(0))

        return output

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size))
        return h0, c0

    def init_hidden2(self, batch_size):
        h0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size//4))
        c0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size//4))
        return h0, c0

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)
