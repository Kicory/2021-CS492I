import torch
import torch.nn as nn

class BaseFC(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        self.allLabel = False

        super(BaseFC, self).__init__()
        self.flatter = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.act(self.dropout(self.bn1(self.fc1(self.flatter(x)))))
        output = self.fc2(output)
        return output

class Classifier(BaseFC):
    def __init__(self):
        self.lr = 0.0001
        self.weight_decay = 0.05
        self.lrStep = 40
        self.lrGamma = 0.9
        super().__init__(128 * 10, 1, 0.5)