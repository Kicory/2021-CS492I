import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Classifier, self).__init__()

        self.allLabel = True

        self.flatter = nn.Flatten()
        self.dropout = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 527)
        self.act = nn.ReLU()

    def forward(self, x):
        output = self.dropout(self.act(self.bn1(self.fc1(self.flatter(x)))))
        output = self.dropout(self.act(self.bn2(self.fc2(output))))
        output = self.fc3(output)
        return output