import torch
import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Conv1D, self).__init__()

        self.flatter = nn.Flatten()
        self.conv1d1 = nn.Conv1d(in_channels, 64, 1)
        self.conv1d2 = nn.Conv1d(64, 32, 1)
        self.fc1 = nn.Linear(10 * 32, 32)
        self.fc2 = nn.Linear(32, out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.transpose(x, 1, 2)
        output = self.flatter(self.dropout(self.act(self.conv1d2(self.dropout(self.act(self.conv1d1(output)))))))
        output = self.fc2(self.dropout(self.act(self.fc1(output))))
        return output

class Classifier(Conv1D):
    def __init__(self):
        self.allLabel = False
        self.lr = 0.0001
        super().__init__(128, 1, 0.2)