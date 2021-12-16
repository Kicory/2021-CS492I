import torch
import torch.nn as nn

class RNNnet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, drop_prob=0.2, bidirectional=True):
        super(RNNnet, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        
        self.relu = nn.ReLU()

        self.rnn = nn.RNN(in_channels, hid_channels, num_layers, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.fc = nn.Linear(hid_channels, out_channels)
            
    def forward(self, x):
        # x.shape = NUM_BATCHES x 10 x 128
        
        out, h = self.rnn(x)
        # Only use the last output
        out = out[:, -1, :].squeeze()
        out = self.fc(self.relu(out))
        return out
    
class Classifier(RNNnet):
    def __init__(self):
        self.lr = 0.0001
        super().__init__(128, 64, 1, 3, drop_prob=0.2, bidirectional=False)
        self.allLabel = False