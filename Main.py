import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import importlib
import matplotlib
import matplotlib.pyplot as plt
import argparse
from AudioSetDataLoader import AudioSetDataSet

parser = argparse.ArgumentParser(description='FC based classifier')

parser.add_argument('--model', '-m', type=str, required=True,
                    help='사용할 모델 종류 (./Models/ 디렉터리 하위의, 모델이 구현된 파이썬 파일 이름) e.g., "-m Base"')

parser.add_argument('--gpu', action='store_true',
                    help='gpu 사용 여부')

parser.add_argument('--train', '-t', action='store_true',
                    help='Training? (Will use default training data in "./RawData/AudioSet/[bal_train, eval]/")')

parser.add_argument('--epoch', '-e', type=int, default=10, required=False,
                    help='epoch 횟수')

parser.add_argument('--evalWith', type=str, required=False, default=None,
                    help='Evaluation에 사용할 features (./RawData 하위의 주소만) e.g., \"-e RecordFeatures\"')

args = parser.parse_args()

module = importlib.import_module(f'Models.{args.model}')
modelSaveDir = f'./TrainedModels/{args.model}/'

trainDataDir = './Dataset/RawData/AudioSet/bal_train/'
validationDataDir = './DataSet/RawData/AudioSet/eval/'
if args.evalWith:
    resultDir = modelSaveDir
    evalDataDir = os.path.join('./RawData', args.evalWith, '').replace("\\","/")
else:
    evalDataDir = None

device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

def train(model, trainingDataLoader, validationDataLoader, optimizer, scheduler, lossFunction):
    stepCount = 0
    bestAcc = 0

    for epoch in range(args.epoch):
        # Here starts the train loop.
        model.train()
        for x, y in trainingDataLoader:
            
            stepCount += 1

            #  Send `x` and `y` to either cpu or gpu using `device` variable. 
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)


            # Feed `x` into the network, get an output, and keep it in a variable called `logit`. 
            logit = model(x)

            # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
            loss = lossFunction(logit, y)
            print('\t' + str(loss.item()).zfill(10), end='\r')
            # flush out the previously computed gradient.
            optimizer.zero_grad()

            # backward the computed loss. 
            loss.backward()

            # update the network weights. 
            optimizer.step()

        # Here starts the test loop.
        model.eval()
        with torch.no_grad():
            test_loss = 0.
            test_accuracy = 0.
            test_num_data = 0.
            for x, y in validationDataLoader:

                # Send `x` and `y` to either cpu or gpu using `device` variable..
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                # Feed `x` into the network, get an output, and keep it in a variable called `logit`.
                logit = model(x)

                # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
                loss = lossFunction(logit, y)

                # Compute accuracy of this batch using `logit`, and keep it in a variable called 'accuracy'.
                accuracy = ((logit > 0) == y).sum()

                test_loss += loss.item() * x.shape[0]
                test_accuracy += accuracy.item()
                test_num_data += x.shape[0]

            test_loss /= test_num_data
            test_accuracy /= test_num_data

            # Just for checking progress
            print(f'Test result of epoch {epoch}/{args.epoch} || loss : {test_loss:.3f} acc : {test_accuracy:.3f} ')

            # Whenever `test_accuracy` is greater than `best_accuracy`, save network weights with the filename 'best.pt' in the directory specified by `ckpt_dir`.
            if test_accuracy > bestAcc:
                bestAcc = test_accuracy
                torch.save(model.state_dict(), os.path.join(modelSaveDir, f'{time.strftime("%H-%M-%S", time.localtime())}_best.pt'))

        scheduler.step()



net = module.Classifier(128 * 10)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.5)

tdl = DataLoader(AudioSetDataSet(trainDataDir, only10Len=True), batch_size=10)
vdl = DataLoader(AudioSetDataSet(validationDataDir, only10Len=True), batch_size=10)
train(net, tdl, vdl, optimizer=optimizer, scheduler=scheduler, lossFunction=nn.BCEWithLogitsLoss())