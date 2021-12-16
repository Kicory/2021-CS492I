import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, bias=True, drop_prob=0.2, bidirectional=False):
        super(LSTMNet, self).__init__()

        
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(in_channels, hid_channels, num_layers, bias=bias, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.fc = nn.Linear(hid_channels * (2 if bidirectional else 1), out_channels)

    def forward(self, x):
        
        out, (h, c) = self.lstm(x)
        # Only use the last output
        out = out[:, -1, :].squeeze()
        out = self.fc(self.relu(out))

        return out
    
class Classifier(LSTMNet):
    def __init__(self):
        self.lr = 0.0001
        super().__init__(128, 64, 1, 2, bias=False, drop_prob=0.2, bidirectional=True)
        self.allLabel = False