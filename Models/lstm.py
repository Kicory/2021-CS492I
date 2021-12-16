import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """
    Usage:

    lstm = LSTMNet(in_channels, hid_channels, out_channels, num_layers, drop_prob=0.2)
    pred = lstm(x)
    음성 파일의 길이가 서로 다르기 때문에 recurrent 한 모델을 사용해야함. 따라서 GRU를 사용함.

    """
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, bias=True, drop_prob=0.2, bidirectional=False):
        super(LSTMNet, self).__init__()

        # self.in_channels = in_channels
        # self.hid_channels = hid_channels
        # self.num_layers = num_layers
        
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(in_channels, hid_channels, num_layers, bias=bias, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.fc = nn.Linear(hid_channels * (2 if bidirectional else 1), out_channels)

    # def _get_init_states(self, batch_size):
    #     weight = next(self.parameters()).data
    #     init_states = weight.new(self.num_layers, batch_size, self.hid_channels).zero_()
    #     return init_states
            
    def forward(self, x):
        # x.shape = NUM_BATCHES x 10 x 128

        # batch_size, time_length, num_features = x.shape
        # init_states = self._get_init_states(batch_size)
        # out, h = self.lstm(x, init_states)
        
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