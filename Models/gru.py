import torch
import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        
        self.relu = nn.ReLU()
        self.gru = nn.GRU(in_channels, hid_channels, num_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hid_channels, out_channels)

    def forward(self, x):
        out, _ = self.gru(x)
        # Only use the last output
        out = out[:, -1, :].squeeze()
        out = self.fc(self.relu(out))
        
        return out
    
class Classifier(GRUNet):
    def __init__(self):
        self.lr = 0.0001
        super().__init__(128, 64, 1, 3, drop_prob=0.2)
        self.allLabel = False