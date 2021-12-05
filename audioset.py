import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from params import FULL_LABEL_COUNT

tqdmWidth = int(os.get_terminal_size().columns / 1.5)

class AudioSetDataSet(Dataset):
    def __init__(self, dir, only10Len=False, allLabel=False):
        self.dir = dir
        self.data = []
        files = os.listdir(dir)
        self.totalLen = len(files)
        self.trueRatio = 0.
        for d in tqdm(files, ncols=tqdmWidth, desc=f"Loading Data from {dir}"):
            for x, y in torch.load(os.path.join(dir, d)):
                if allLabel:
                    y = torch.zeros(FULL_LABEL_COUNT).scatter_(0, y, 1)
                else:
                    isLaugh = 16 in y or 18 in y or 20 in y or 21 in y
                    y = torch.Tensor([isLaugh])
                    self.trueRatio += 1. if isLaugh else 0
                
                if (not only10Len) or x.size()[0] == 10:
                    self.data.append((x, y))
        if allLabel:
            self.trueRatio = None
        else:
            self.trueRatio /= (self.__len__() - self.trueRatio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def getTrueRatio(self):
        return self.trueRatio