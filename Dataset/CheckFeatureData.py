import torch
import argparse
import os
import time
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='녹음으로부터 추출된 feature data 갖고 노는 용도의 프로그램')

parser.add_argument('--dataDir', '-d', type=str, required=True,
                    help='VGGish를 통해 추출된 feature가 저장된 디렉터리 (RawData 하위의) (e.g., \"-d features\"')

args = parser.parse_args()

dataDir = os.path.join('./RawData', args.dataDir, '').replace("\\","/")

device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
data = []

for ptFile in os.listdir(dataDir):
    data.append(torch.load(dataDir + ptFile).to(device))

data = torch.stack(data, dim=0)

plt.plot(range(data.size()[0]), data[:, :, 1].squeeze().numpy())
plt.show()
plt.plot(range(data.size()[0]), data[:, :, 2].squeeze().numpy())
plt.show()
plt.plot(range(data.size()[0]), data[:, :, 3].squeeze().numpy())
plt.show()
plt.plot(range(data.size()[0]), data[:, :, 4].squeeze().numpy())
plt.show()