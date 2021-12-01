import torch
import argparse
import os
import sounddevice as sd
from scipy.io.wavfile import write
from threading import Timer
import time

parser = argparse.ArgumentParser(description='녹음 및 VGGish를 이용한 feature extraction 자동 수행')

parser.add_argument('--wavOutDir', type=str, required=True,
                    help='녹음된 wav파일이 저장될 디렉터리 (RawData 아래) \ne.g.,\"--wavOutDir recordedFiles\"')

parser.add_argument('--outDir', '-f', type=str, required=True,
                    help='VGGish를 통해 추출된 feature가 저장될 디렉터리 (RawData 아래) \ne.g., \"-f features\"')

parser.add_argument('--count','-n', type=int, required=True,
                    help='녹음할 wav 파일 갯수 (= 만들어질 feature 파일 갯수)')

parser.add_argument('--len', '-l', type=int, default=10, required=False,
                    help='각 wav파일의 길이. 기본 10초')

parser.add_argument('--stride', '-s', type=float, required=False, default=0,
                    help='녹음 파일 간 시간 간격')

args = parser.parse_args()

wavOutDir = os.path.join('./RawData', args.wavOutDir, '').replace("\\","/")
outDir = os.path.join('./RawData', args.outDir, '').replace("\\","/")
count = args.count
wavLen = args.len
wavStride = args.stride

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

def doMainUnitWork(model, index):
    outWavFileName = str(index).zfill(4) + '.wav'
    wavFilePath = recordFile(outWavFileName)
    torch.save(model.forward(wavFilePath).detach(), outDir + str(index).zfill(4) + '.pt')

def looper(model, workCount):
    if not os.path.exists(wavOutDir):
        os.makedirs(wavOutDir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    index = 0
    while index < workCount:
        start_time = time.time()
        if index + 1 == workCount:
            doMainUnitWork(model, index)
            printProgress(index, workCount, title="{:4d}/{:4d}".format(index + 1, workCount))
            break
        else:
            recTask = Timer(wavStride, doMainUnitWork, args=(model, index, ))
            recTask.daemon = True
            recTask.start()

        printProgress(index, workCount, title="{:4d}/{:4d}".format(index + 1, workCount))

        index = index + 1
        time.sleep(wavStride + wavLen - (time.time() - start_time))

def main():
    model = getVGGishModel()
    looper(model, count)

def printProgress(index, total, bar_len=50, title='Please wait'):

    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

if __name__ == '__main__':
    main()