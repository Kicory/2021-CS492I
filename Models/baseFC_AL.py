import torch
import torch.nn as nn
from .baseFC import BaseFC
from params import FULL_LABEL_COUNT

class Classifier(BaseFC):
    def __init__(self):
        super().__init__(128 * 10, FULL_LABEL_COUNT, 0.2)
        self.allLabel = True