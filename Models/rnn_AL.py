import torch
from .rnn import RNNnet
from params import FULL_LABEL_COUNT
    
class Classifier(RNNnet):
    def __init__(self):
        super().__init__(128, 128, FULL_LABEL_COUNT, 3, drop_prob=0.2, bidirectional=False)
        self.allLabel = True