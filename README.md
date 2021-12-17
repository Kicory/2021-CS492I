
# 2021-CS492I Final Project: Team 25 - OLA

**OLA: Office Laughter Analyzer** is an audio-based real-time office laughter detection system.

## `main.py`
`main.py` does training on [AudioSet](https://research.google.com/audioset/)'s feature data, and test the f1 score with evaluation data each epoch.
### Usage
`python main.py -m "model file name(without .py)" [--gpu] [--roll]`

*Example:* `python main.py -m dense --gpu --roll`

`-m / --model`: Model to use (filename in directory `./Models/`

`--gpu`: Use gpu to training.

`--roll`: Use Embedding Rolling to augment input data. The data will be augmented by 10 times.

### Other Settings

*Example:* `python main.py -m dense --noise 5 --trueWeight 0.2 --positiveWeight 1 --batch_size 256 --epoch 10`

`--noise N`: Use noise augmentation. The data will be augmented by N times. **Proven to be useless. Don't use.**

`--trueWeight W`: Use weighted training, and set **true labeled case** weight to W. Note that weight value **'1.0' means ~100 times stronger gradient** compared to **True Negative** cases.

`--positiveWeight W`: Use weighted training, and set **positive output case** weight to W. Note that weight value **'1.0' means ~100 times stronger gradient** compared to **True Negative** cases.

`--batch_size B`: Set the batch size B.

`--epoch E`: Set the epoch E.

## `live.py`
`live.py` records audio with local microphone device, and saves the `.wav` file, and extract embedding vector using VGGish model downloaded, and do laughter inference with specified model.

### Usage
`python live.py -m "model file name(without .py)" --wavOutDir FILE_DIR --count N`

*Example:* `python live.py -m dense --wavOutDir recordedWavs --count 10`

`-m / --model`: Model to use (filename in directory `./Models/`

`--wavOutDir PATH`: Where to save recorded audio file (`.wav`). The audio files will be saved in `./Dataset/RawData/[PATH]`

`--count N`: How many files to record? Note that one audio file has 10 seconds long.
