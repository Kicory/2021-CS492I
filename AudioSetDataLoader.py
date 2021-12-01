import torch
import os
from torch.utils.data import Dataset

class AudioSetDataSet(Dataset):
    def __init__(self, dir, only10Len=False):
        self.dir = dir
        self.data = []
        files = os.listdir(dir)
        self.totalLen = len(files)
        for index, d in enumerate(files):
            for x, y in torch.load(os.path.join(dir, d)):
                y = torch.Tensor([16 in y or 20 in y])
                if only10Len and x.size()[0] == 10:
                    self.data.append((x, y))

            printProgress(index, self.totalLen, title="Loading Data...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def printProgress(index, total, bar_len=50, title='Please wait'):

    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')