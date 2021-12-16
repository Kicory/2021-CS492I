import os
from typing import Tuple
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader, dataset
import torch.optim as optim
import importlib
import matplotlib.pyplot as plt
import argparse
from audioset import AudioSetDataSet
from tqdm import tqdm

parser = argparse.ArgumentParser(description='모델 train수행')

parser.add_argument('--model', '-m', type=str, required=True,
                    help='사용할 모델 종류 (./Models/ 디렉터리 하위의, 모델이 구현된 파이썬 파일 이름) e.g., "-m Base"')

parser.add_argument('--gpu', action='store_true',
                    help='gpu 사용 여부')

parser.add_argument('--roll', action='store_true',
                    help='input tensor의 앞을 뒤로 보내는 방식으로 augment')

parser.add_argument('--noise', type=int, required=False, default=0,
                    help='임베딩에 조금의 noise를 넣어서 augment')

parser.add_argument('--trueWeight', type=float, required=False, default=None,
                    help='weights[y == 1] = [trueWeight]')

parser.add_argument('--positiveWeight', type=float, required=False, default=None,
                    help='weights[logit > 0] = [positiveWeight]')

parser.add_argument('--trainData', type=str, required=False, default='bal_train',
                    help='Training? (Will use default training data in ".Dataset/RawData/AudioSet/bal_train/")')

parser.add_argument('--batch_size', type=int, required=False, default=10,
                    help='배치 사이즈')

parser.add_argument('--epoch', '-e', type=int, default=10, required=False,
                    help='epoch 횟수')

args = parser.parse_args()

# import model to train
module = importlib.import_module(f'Models.{args.model}')

modelSaveDir = './TrainedModels/'

trainDataDir = f'./Dataset/RawData/AudioSet/{args.trainData}/'
validationDataDir = './DataSet/RawData/AudioSet/eval/'
roll = args.roll
noise = args.noise

tqdmWidth = int(os.get_terminal_size().columns / 1.5)


def doForward(x, y, model, device, lossFunction):
    #  Send `x` and `y` to either cpu or gpu using `device` variable. 
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)

    # Feed `x` into the network, get an output, and keep it in a variable called `logit`. 
    logit = model(x)
    # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
    loss = lossFunction(logit, y)

    return x, y, logit, loss

def train(model, datasets: Tuple[AudioSetDataSet, AudioSetDataSet], optimizer, scheduler, lossFunction):
    tds, vds = datasets
    tdl = DataLoader(tds, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    vdl = DataLoader(vds, batch_size=args.batch_size, pin_memory=True)

    best_f1 = 0
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    model = model.to(device)

    for epoch in range(args.epoch):
        # Here starts the train loop.
        model.train()
        trainLossSum = 0
        trainStepCount = 0
        for xRaw, yRaw in tqdm(tdl, desc=f"training... {str(epoch + 1).zfill(3)} / {str(args.epoch).zfill(3)}", ncols=tqdmWidth, leave=False):
            
            _, y, logit, loss = doForward(xRaw, yRaw, model, device, lossFunction)
            trainLossSum += loss.mean().item()
    
            if model.allLabel:
                # Loss weight: False인 classes vs. True인 classes가 gradient에 기여하는 비율 동일하게 설정
                totalTrueLabel = y.sum().item()
                weightRatio = totalTrueLabel / (y.numel() - totalTrueLabel)
                weights = torch.full_like(y, weightRatio)
                weights[y == 1] = 1
                loss = (loss * weights).sum()
            else:
                # Loss weight: False인 경우(웃음 X) vs. True인 경우(웃음 O)가
                # 전체 training 과정에서 기여하는 비율을 동일하게 설정
                weights = torch.full_like(y, tds.trueRatio)
                if args.trueWeight is not None:
                    weights[y == 1] = args.trueWeight
                # weights[(logit > 0) == (y == 0)] = 1
                # weights[(logit > 0) == (y == 0)] = tds.trueRatio
                if args.positiveWeight is not None:
                    weights[logit > 0] = args.positiveWeight
                loss = (loss * weights).sum()

            # flush out the previously computed gradient.
            optimizer.zero_grad()

            # backward the computed loss.
            loss.backward()

            # update the network weights. 
            optimizer.step()
            trainStepCount += 1

        # Here starts the test loop.
        model.eval()
        with torch.no_grad():
            test_loss = 0.
            test_trues = 0.
            test_num_true_positive = 0.
            test_num_data = 0.
            test_num_positive = 0.
            test_num_trueLabel = 0.
            logit_absolute_sum = 0.
            for xRaw, yRaw in tqdm(vdl, desc=f"validating... ", ncols=tqdmWidth, leave=False):

                xRaw, yRaw, logit, loss = doForward(xRaw, yRaw, model, device, lossFunction)
                # weighting으로 reduction을 none으로 설정해 놓았기 때문에, 여기 (eval) 에서는 mean()을 걸어 준다.
                loss = loss.mean()
                logit_absolute_sum += logit.abs().sum()

                # Compute TP, TN, FP, FN
                _true_postive = ((logit > 0) * yRaw).sum().item()
                _true_negative = ((logit <= 0) * (yRaw == 0)).sum().item()
                _false_positive = ((logit > 0) * (yRaw == 0)).sum().item()
                _false_negative = ((logit <= 0) * yRaw).sum().item()
                
                data_count = torch.numel(logit)

                test_loss += loss.item() * data_count
                test_trues += _true_postive + _true_negative
                test_num_true_positive += _true_postive
                test_num_data += data_count
                test_num_positive += _true_postive + _false_positive
                test_num_trueLabel += _true_postive + _false_negative

            test_loss /= test_num_data

            accuracy = test_trues / test_num_data
            if test_num_true_positive == 0:
                precision = 0
            else:
                precision = test_num_true_positive / test_num_positive

            if test_num_true_positive == 0:
                recall = 0
            else:
                recall = test_num_true_positive / test_num_trueLabel
            
            if recall == 0 or precision == 0:
                f1_score = 0
            else:
                f1_score = 2 * (recall * precision) / (recall + precision)

            # Just for checking progress
            print(f'Result of epoch {str(epoch + 1).zfill(2)}/{str(args.epoch).zfill(2)} || logit mean: {logit_absolute_sum / test_num_data:.3f} train loss: {trainLossSum / trainStepCount:.3f} loss: {test_loss:.3f} acc: {accuracy:.3f} precision: {precision:.3f} recall: {recall:.3f} F1 score: {f1_score:.3f}')
            print(f'Other Metrics || Total: {int(test_num_data):d} Total Positives: {int(test_num_positive):d} Total Trues: {int(test_num_trueLabel):d} Max F1: {best_f1:.3f}')

            # Whenever `test_accuracy` is greater than `best_accuracy`, save network weights with the filename 'best.pt' in the directory specified by `ckpt_dir`.
            if f1_score > best_f1:
                best_f1 = f1_score
                if not os.path.exists(modelSaveDir):
                    os.mkdir(modelSaveDir)
                torch.save(model.state_dict(), os.path.join(modelSaveDir, f'{args.model}_best.pt'))
        scheduler.step()

net = module.Classifier()
net.train()
optimizer = optim.AdamW(net.parameters(), lr=getattr(net, 'lr', 0.001), weight_decay=getattr(net, 'weight_decay', 0.05))
scheduler = scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=getattr(net, 'lrStep', 20), gamma=getattr(net, 'lrGamma', 0.75))
tds = AudioSetDataSet(trainDataDir, only10Len=True, allLabel=net.allLabel, roll=roll, noise=noise)
vds = AudioSetDataSet(validationDataDir, only10Len=True, allLabel=net.allLabel, roll=False, noise=False)

train(net, (tds, vds), optimizer=optimizer, scheduler=scheduler, lossFunction=BCEWithLogitsLoss(reduction='none'))