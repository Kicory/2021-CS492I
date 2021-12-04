import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import importlib
import matplotlib.pyplot as plt
import argparse
from audioset import AudioSetDataSet
from tqdm import tqdm

parser = argparse.ArgumentParser(description='모델 train 및 evaluation 수행')

parser.add_argument('--model', '-m', type=str, required=True,
                    help='사용할 모델 종류 (./Models/ 디렉터리 하위의, 모델이 구현된 파이썬 파일 이름) e.g., "-m Base"')

parser.add_argument('--gpu', action='store_true',
                    help='gpu 사용 여부')

parser.add_argument('--trainData', type=str, required=False, default='bal_train',
                    help='Training? (Will use default training data in ".Dataset/RawData/AudioSet/bal_train/")')

parser.add_argument('--batch_size', type=int, required=False, default=10,
                    help='배치 사이즈')

parser.add_argument('--epoch', '-e', type=int, default=10, required=False,
                    help='epoch 횟수')

parser.add_argument('--evalWith', type=str, required=False, default=None,
                    help='Evaluation에 사용할 features (./RawData 하위의 주소만) e.g., \"-e RecordFeatures\"')

args = parser.parse_args()

# import model to train & evaluate
module = importlib.import_module(f'Models.{args.model}')

modelSaveDir = f'./TrainedModels/{args.model}/'

trainDataDir = f'./Dataset/RawData/AudioSet/{args.trainData}/'
validationDataDir = './DataSet/RawData/AudioSet/eval/'

if args.evalWith:
    resultDir = modelSaveDir
    evalDataDir = os.path.join('./RawData', args.evalWith, '').replace("\\","/")
else:
    evalDataDir = None

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

def train(model, trainingDataLoader, validationDataLoader, optimizer, scheduler, lossFunction):
    best_f1 = 0
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    model = model.to(device)

    for epoch in range(args.epoch):
        # Here starts the train loop.
        model.train()
        for x, y in tqdm(trainingDataLoader, desc=f"training... {epoch} / {args.epoch}", ncols=tqdmWidth, leave=False):
            
            x, y, _, loss = doForward(x, y, model, device, lossFunction)

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
            test_trues = 0.
            test_num_true_positive = 0.
            test_num_data = 0.
            test_num_positive = 0.
            test_num_trueLabel = 0.
            for x, y in tqdm(validationDataLoader, desc=f"validating... ", ncols=tqdmWidth, leave=False):

                x, y, logit, loss = doForward(x, y, model, device, lossFunction)

                # Compute TP, TN, FP, FN
                _true_postive = ((logit > 0) * y).sum().item()
                _true_negative = ((logit <= 0) * (y == 0)).sum().item()
                _false_positive = ((logit > 0) * (y == 0)).sum().item()
                _false_negative = ((logit <= 0) * y).sum().item()
                
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
            print(f'Test result of epoch {epoch + 1}/{args.epoch} || loss: {test_loss:.3f} acc: {accuracy:.3f} precision: {precision:.3f} recall: {recall:.3f} F1 score: {f1_score:.3f}')

            # Whenever `test_accuracy` is greater than `best_accuracy`, save network weights with the filename 'best.pt' in the directory specified by `ckpt_dir`.
            if f1_score > best_f1:
                best_f1 = f1_score
                if not os.path.exists(modelSaveDir):
                    os.mkdir(modelSaveDir)
                torch.save(model.state_dict(), os.path.join(modelSaveDir, f'{time.strftime("%H-%M-%S", time.localtime())}_best.pt'))

        scheduler.step()


net = module.Classifier()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.5)

tdl = DataLoader(AudioSetDataSet(trainDataDir, only10Len=True, allLabel=net.allLabel), batch_size=args.batch_size, pin_memory=True)
vdl = DataLoader(AudioSetDataSet(validationDataDir, only10Len=True, allLabel=net.allLabel), batch_size=args.batch_size, pin_memory=True)


train(net, tdl, vdl, optimizer=optimizer, scheduler=scheduler, lossFunction=nn.BCEWithLogitsLoss())