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
                    help='input tensor의 앞을 뒤로 보내는 방식으로 10배 augment (사용할 것)')

parser.add_argument('--noise', type=int, required=False, default=0,
                    help='임베딩에 조금의 noise를 넣어서 augment하는 방식. 0부터 정수값을 넣으면 하나의 input 당 N개 augment (사용하지 말 것)')

parser.add_argument('--trueWeight', type=float, required=False, default=0.2,
                    help='Weighted Training: Weight for true label. i.e., weights[y == 1] = [trueWeight]')

parser.add_argument('--positiveWeight', type=float, required=False, default=1,
                    help='Weighted Training: Weight for positive output. i.e., weights[logit > 0] = [positiveWeight]')

parser.add_argument('--batch_size', type=int, required=False, default=256,
                    help='배치 사이즈')

parser.add_argument('--epoch', '-e', type=int, default=100, required=False,
                    help='epoch 횟수')

parser.add_argument('--eval', action='store_true',
                    help='Training 없이, 저장된 모델 데이터로 evaluation만 진행')

args = parser.parse_args()

# import model to train
module = importlib.import_module(f'Models.{args.model}')

modelSaveDir = './TrainedModels/'
modelSaveFile = os.path.join(modelSaveDir, f'{args.model}_best.pt')

trainDataDir = f'./Dataset/RawData/AudioSet/bal_train/'
validationDataDir = './DataSet/RawData/AudioSet/eval/'
roll = args.roll
noise = args.noise
onlyEval = args.eval

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
    if os.path.isfile(modelSaveFile):
        # Load previous train result (to not earse better saved model)
        best_f1 = torch.load(modelSaveFile)['f1']
    
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
    
            # Weighted Training
            weights = torch.full_like(y, tds.trueRatio)
            if args.trueWeight is not None:
                weights[y == 1] = args.trueWeight
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
            best_f1 = test_model(vdl, model, device, lossFunction, epoch, best_f1)
        scheduler.step()

def test_model(vdl, model, device, lossFunction, epoch, best_f1):
    test_loss = 0.
    test_trues = 0.
    test_num_true_positive = 0.
    test_num_data = 0.
    test_num_positive = 0.
    test_num_trueLabel = 0.
    logit_absolute_sum = 0.
    for xRaw, yRaw in tqdm(vdl, desc="validating... ", ncols=tqdmWidth, leave=False):

        xRaw, yRaw, logit, loss = doForward(xRaw, yRaw, model, device, lossFunction)
        # weighting으로 reduction을 none으로 설정해 놓았기 때문에, 여기 (eval) 에서는 mean()을 걸어 준다.
        loss = loss.mean()
        logit_absolute_sum += logit.abs().sum()

        # Compute TP, TN, FP, FN to calculate F1 score!
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

    
    if onlyEval:
        print(f'Result || loss: {test_loss:.3f} acc: {accuracy:.3f} precision: {precision:.3f} recall: {recall:.3f} F1 score: {f1_score:.3f}')
    else:
        print(f'Result of epoch {str(epoch + 1).zfill(2)}/{str(args.epoch).zfill(2)} || logit mean: {logit_absolute_sum / test_num_data:.3f} loss: {test_loss:.3f} acc: {accuracy:.3f} precision: {precision:.3f} recall: {recall:.3f} F1 score: {f1_score:.3f}')

    if not onlyEval and (f1_score > best_f1):
        best_f1 = f1_score
        if not os.path.exists(modelSaveDir):
            os.mkdir(modelSaveDir)
        torch.save({'f1': best_f1, 'dict': model.state_dict()}, modelSaveFile)

    if not onlyEval:
        print(f'Other Metrics || Total: {int(test_num_data):d} Total Positives: {int(test_num_positive):d} Total Trues: {int(test_num_trueLabel):d} Max F1: {best_f1:.3f}')
        
    return best_f1
    

"""
main.py code
"""
# Make imported model
net = module.Classifier()

# Load validation data (balanced test)
vds = AudioSetDataSet(validationDataDir, only10Len=True, roll=False, noise=False)


if not onlyEval:
    net.train()
    optimizer = optim.AdamW(net.parameters(), lr=getattr(net, 'lr', 0.001), weight_decay=getattr(net, 'weight_decay', 0.05))
    scheduler = scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=getattr(net, 'lrStep', 20), gamma=getattr(net, 'lrGamma', 0.75))
    tds = AudioSetDataSet(trainDataDir, only10Len=True, roll=roll, noise=noise)
    train(net, (tds, vds), optimizer=optimizer, scheduler=scheduler, lossFunction=BCEWithLogitsLoss(reduction='none'))

if onlyEval:
    vdl = DataLoader(vds, batch_size=args.batch_size, pin_memory=True)
    net.load_state_dict(torch.load(os.path.join('./TrainedModels/', f'{args.model}_best.pt'), map_location=torch.device('cpu'))['dict'])
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        test_model(vdl, net, device, BCEWithLogitsLoss(reduction='none'), None, None)