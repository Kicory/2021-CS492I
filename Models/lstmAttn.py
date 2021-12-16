import torch
import torch.nn as nn

class LSTMAttnNet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, bias=True, drop_prob=0.2, bidirectional=False):
        super(LSTMAttnNet, self).__init__()
        self.lstm = nn.LSTM(in_channels, hid_channels, num_layers, bias=bias, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.attn = nn.Linear(hid_channels * (2 if bidirectional else 1), hid_channels * (2 if bidirectional else 1))
        self.fc = nn.Linear(2 * hid_channels * (2 if bidirectional else 1), out_channels)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        out, (h, c) = self.lstm(x)
        enc = self.attn(out[:, -1, :].squeeze())
        attn = torch.softmax(torch.matmul(out, enc.unsqueeze(-1)), dim=-1) * out
        context = torch.sum(attn, dim=1)
        out = self.fc(self.relu(torch.concat((context, enc), dim=1)))

        return out
    
class Classifier(LSTMAttnNet):
    def __init__(self):
        self.lr = 0.001
        self.weight_decay = 0.05
        self.lrStep = 10
        self.lrGamma = 0.9
        super().__init__(128, 64, 1, 2, bias=False, drop_prob=0.2, bidirectional=True)
        self.allLabel = False