import torch
import torch.nn as nn
from .lstm import LSTMNet
from params import FULL_LABEL_COUNT
    
class Classifier(LSTMNet):
    def __init__(self):
        super().__init__(128, 128, FULL_LABEL_COUNT, 3, bias=False, drop_prob=0.2, bidirectional=False)
        self.allLabel = True