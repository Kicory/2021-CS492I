import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from params import FULL_LABEL_COUNT

tqdmWidth = int(os.get_terminal_size().columns / 1.5)

class AudioSetDataSet(Dataset):
    def __init__(self, dir, only10Len=False, allLabel=False, roll=False, noise=0):
        self.dir = dir
        self.data = []
        files = os.listdir(dir)
        self.totalLen = len(files)
        self.trueRatio = 0.
        self.trueSwitch = False
        for d in tqdm(files, ncols=tqdmWidth, desc=f"Loading Data from {dir}"):
            for x, y in torch.load(os.path.join(dir, d)):

                embedLen, featureDim = x.size()

                if allLabel:
                    y = torch.zeros(FULL_LABEL_COUNT).scatter_(0, y, 1)
                else:
                    isLaugh = 16 in y or 18 in y or 20 in y or 21 in y
                    y = torch.Tensor([isLaugh])
                    self.trueSwitch = isLaugh
                
                toAdd = []

                # 노이즈 없는 순수 Input을 넣어줌
                toAdd.append((x, y))

                # 앞뒤가 약간씩 바뀌어도 웃는 소리는 웃는 소리임!!!
                # 굴려서 죄다 넣음
                if roll:
                    for rollAmt in range(embedLen):
                        toAdd.append((torch.roll(x, rollAmt, 0), y))
                    else:
                        toAdd.append((x, y))

                # noise 값이 1 이상일 때만 noise 값만큼 반복함
                for _ in range(noise):
                    x_noise = self.getNoised(x, 0.1, 5)

                    #노이즈 추가된 input도 전부 굴려서 넣어줌 (뻥튀기)
                    if roll:
                        for rollAmt in range(embedLen):
                            toAdd.append((torch.roll(x_noise, rollAmt, 0), y))
                    else:
                        toAdd.append((x_noise, y))
                
                if (not only10Len) or embedLen == 10:
                    if self.trueSwitch:
                        self.trueRatio += len(toAdd)
                    self.data += toAdd
        if allLabel:
            self.trueRatio = None
        else:
            self.trueRatio /= (self.__len__() - self.trueRatio)
        print(f"true 라벨 비율: {self.trueRatio}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def getNoised(self, x, prob=0.2, noiseRange=1):
        maxSeed = 1000
        halfProb = prob / 2.
        rand = torch.randint_like(x, low=0, high=maxSeed, dtype=torch.int16)
        rand = torch.where(rand <= (maxSeed * prob), random.randint(-noiseRange, noiseRange), 0)
        return x + rand
