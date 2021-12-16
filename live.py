import importlib
import torch
import argparse
import os
import sounddevice as sd
from scipy.io.wavfile import write
from threading import Timer
import time

parser = argparse.ArgumentParser(description='녹음 및 VGGish를 이용한 feature extraction + 모델을 이용한 inference 자동 수행')

parser.add_argument('--wavOutDir', type=str, required=True,
                    help='녹음된 wav파일이 저장될 디렉터리 (RawData 아래) e.g.,\"--wavOutDir recordedFiles\"')

parser.add_argument('--outDir', '-f', type=str, required=False, default=None,
                    help='VGGish를 통해 추출된 feature가 저장될 디렉터리 (RawData 아래) 지정하지 않으면 저장되지 않음. e.g., \"-f features\"')

parser.add_argument('--model', '-m', type=str, required=False, default=None,
                    help='사용할 모델 종류 (./Models/ 디렉터리 하위의, 모델이 구현된 파이썬 파일 이름) e.g., "-m Base"')

parser.add_argument('--count','-n', type=int, required=True,
                    help='녹음할 wav 파일 갯수')

parser.add_argument('--stride', '-s', type=float, required=False, default=0,
                    help='녹음 간 시간 간격')

args = parser.parse_args()

wavOutDir = os.path.join('./Dataset/RawData', args.wavOutDir, '').replace("\\","/")
if args.outDir is not None:
    outDir = os.path.join('./Dataset/RawData', args.outDir, '').replace("\\","/")
else:
    outDir = None

labelList = torch.load('labelList.pt')

if args.model is not None:
    # import model to eval
    module = importlib.import_module(f'Models.{args.model}')
    model = module.Classifier()
    model.load_state_dict(torch.load(os.path.join('./TrainedModels/', f'{args.model}_best.pt'), map_location=torch.device('cpu')))
    model.eval()
else:
    model = None

count = args.count
wavLen = 10
wavStride = args.stride

tqdmWidth = int(os.get_terminal_size().columns / 1.5)

def getVGGishModel():
    model = torch.hub.load('harritaylor/torchvggish', 'vggish', verbose=False)
    model.eval()
    return model

def recordFile(wavFileName):
    fs = 16000 #16Khz
    r = sd.rec(int(wavLen * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    outWavFile = wavOutDir + wavFileName
    write(outWavFile, fs, r)
    return outWavFile

def doMainUnitWork(embedder, index):
    outWavFileName = str(index).zfill(4) + '.wav'
    wavFilePath = recordFile(outWavFileName)
    embed = embedder.forward(wavFilePath).detach()

    if outDir is not None:
        torch.save(embed, outDir + str(index).zfill(4) + '.pt')
    
    if model is not None:
        result = model(embed.to(device='cpu').unsqueeze(0)).detach().squeeze()
        desc = "... is laughter!" if result.item() > 0 else "... is not laughter."
        print(str(index).zfill(4), desc, f": {result.item():.3f}")


def looper(embedder, workCount):
    if not os.path.exists(wavOutDir):
        os.makedirs(wavOutDir)
    if outDir is not None and not os.path.exists(outDir):
        os.makedirs(outDir)

    index = 0
    for index in range(workCount):
        start_time = time.time()
        recTask = Timer(wavStride, doMainUnitWork, args=(embedder, index, ))
        recTask.daemon = True
        recTask.start()
        time.sleep(wavStride + wavLen - (time.time() - start_time))
    time.sleep(5)


def main():
    embedder = getVGGishModel()
    looper(embedder, count)

if __name__ == '__main__':
    main()