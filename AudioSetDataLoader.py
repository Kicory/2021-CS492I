import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

tqdmWidth = int(os.get_terminal_size().columns / 1.5)

class AudioSetDataSet(Dataset):
    def __init__(self, dir, only10Len=False, allLabel=False):
        self.dir = dir
        self.data = []
        files = os.listdir(dir)
        self.totalLen = len(files)
        for d in tqdm(files, ncols=tqdmWidth, desc=f"Loading Data from {dir}"):
            for x, y in torch.load(os.path.join(dir, d)):
                if allLabel:
                    y = torch.zeros(527).scatter_(0, y, 1)
                else:
                    y = torch.Tensor([16 in y or 18 in y or 20 in y or 21 in y])
                
                if (not only10Len) or x.size()[0] == 10:
                    self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]