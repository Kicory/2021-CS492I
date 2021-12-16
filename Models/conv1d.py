import torch
import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Conv1D, self).__init__()

        self.flatter = nn.Flatten()
        self.conv1d1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1d1 = nn.BatchNorm1d(64)
        self.conv1d2 = nn.Conv1d(64, 32, 1)
        self.bn1d2 = nn.BatchNorm1d(32)
        self.conv1d3 = nn.Conv1d(32, 16, 1)
        self.bn1d3 = nn.BatchNorm1d(16)
        self.conv1d4 = nn.Conv1d(16, 8, 1)
        self.bn1d4 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(10 * 8, 10)
        self.bn1d5 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, out_channels)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.transpose(x, 1, 2)
        output = self.dropout(self.act(self.bn1d1(self.conv1d1(output))))
        output = self.dropout(self.act(self.bn1d2(self.conv1d2(output))))
        output = self.dropout(self.act(self.bn1d3(self.conv1d3(output))))
        output = self.dropout(self.act(self.bn1d4(self.conv1d4(output))))
        output = self.flatter(output)
        output = self.fc2(self.act(self.bn1d5(self.fc1(output))))
        return output

class Classifier(Conv1D):
    def __init__(self):
        self.allLabel = False
        self.lr = 0.001
        self.weight_decay = 0.01
        self.lrStep = 10
        self.lrGamma = 1
        super().__init__(128, 1, 0.2)