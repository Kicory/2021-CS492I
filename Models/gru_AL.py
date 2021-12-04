import torch
import torch.nn as nn
from .gru import GRUNet
from params import FULL_LABEL_COUNT
    
class Classifier(GRUNet):
    def __init__(self):
        super().__init__(128, 128, FULL_LABEL_COUNT, 3, drop_prob=0.2)
        self.allLabel = True