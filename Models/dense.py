import torch
import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Dense, self).__init__()

        self.flatter = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.act(self.dropout(self.bn1(self.fc1(self.flatter(x)))))
        output = self.act(self.dropout(self.bn2(self.fc2(output))))
        output = self.fc3(output)
        return output

class Classifier(Dense):
    def __init__(self):
        self.allLabel = False
        self.lr = 0.0001
        self.weight_decay = 0.05
        self.lrStep = 10
        self.lrGamma = 0.9

        super().__init__(128 * 10, 1, 0.4)

