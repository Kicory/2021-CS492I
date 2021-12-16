import torch
import torch.nn as nn


class LSTMAttnNet(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, bias=True, drop_prob=0.2, bidirectional=False):
        super(LSTMAttnNet, self).__init__()
        # self.mhattn = nn.MultiheadAttention(128, 8, 0.2, True, batch_first=True)
        self.lstm = nn.LSTM(in_channels, hid_channels, num_layers, bias=bias, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.attn = nn.Linear(hid_channels * (2 if bidirectional else 1), hid_channels * (2 if bidirectional else 1))
        self.fc = nn.Linear(2 * hid_channels * (2 if bidirectional else 1), out_channels)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        # self.fcattn = nn.Linear(10 * in_channels, hid_channels)
        # self.fcattn2 = nn.Linear(10 * in_channels, hid_channels)
        
    def forward(self, x):        
        # b, t, n = x.shape
        # attn, w = self.mhattn(x, x, x)
        # enc = self.fcattn2(x.view(b, t*n)).squeeze()
        # attn = self.fcattn(attn.reshape(b, t*n))
        # attn = self.fc(self.relu(torch.concat(attn, enc)))
        # return attn
        
        out, (h, c) = self.lstm(x)

        # enc = self.fcattn(x.view(b, t*n)).squeeze()
        enc = self.attn(out[:, -1, :].squeeze())
        attn = torch.softmax(torch.matmul(out, enc.unsqueeze(-1)), dim=-1) * out
        context = torch.sum(attn, dim=1)
        out = self.fc(self.relu(torch.concat((context, enc), dim=1)))

        return out
    
class Classifier(LSTMAttnNet):
    def __init__(self):
        self.lr = 0.001
        self.weight_decay = 0.05
        self.lrStep = 10
        self.lrGamma = 0.9
        super().__init__(128, 64, 1, 2, bias=False, drop_prob=0.2, bidirectional=True)
        self.allLabel = False


# 1차 시도: self.fc(self.relu(torch.concat((context, enc), dim=1)))
# 2차 시도: self.fc(self.relu(torch.concat((context, enc), dim=1))) 쪼금 더 좋은듯
# 3차 시도: 어텐션을 lstm output을 쓰지 않고 fullyconnected 로 구성함. 큰 차이 없는듯
# 4차 시도: 