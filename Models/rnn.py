import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, drop_prob=0.2, bidirectional = True):
        super(GRUNet, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        
        self.relu = nn.ReLU()
        
        self.rnn = nn.rnn(in_channels, hid_channels, num_layers, batch_first=True, dropout=drop_prob, bidirectional = True)
        self.fc = nn.Linear(hid_channels, out_channels)
            
    def forward(self, x):
        # x.shape = NUM_BATCHES x 10 x 128
        out = self.fc(self.relu(self.rnn(x)))
        return out
    
