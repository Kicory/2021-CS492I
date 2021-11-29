import torch
from tfrecord.torch.dataset import TFRecordDataset
import os
import argparse
import requests
import tarfile

parser = argparse.ArgumentParser(description='Produce pytorch-friendly .pt files from tfrecords of AudioSet.')
parser.add_argument('--dir', metavar='F', type=str,
                    help='output folder name')
args = parser.parse_args()

url = "http://storage.googleapis.com/asia_audioset/youtube_corpus/v1/features/features.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path=".")
rawFolder = 'audioset_v1_embeddings'

def doSave(folder):
    tfrecord_path = lambda folderName : f"./{folderName}/{folder}/"
    if not os.path.exists(tfrecord_path(args.dir)):
        os.makedirs(tfrecord_path(args.dir))

    description = {"labels": "int"}
    seqDesc = {"audio_embedding": "byte"} 

    idx = 0

    for tfrec in os.listdir(tfrecord_path(rawFolder)):
        data = []
        dataset = TFRecordDataset(tfrecord_path(rawFolder) + tfrec, index_path=None, description=description, sequence_description=seqDesc)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        for label, emb in loader:
            data.append((torch.cat(emb['audio_embedding']), label['labels'].squeeze(0)))
        
        torch.save(data, tfrecord_path(args.dir) + str(idx).zfill(4) + '.pt')
        idx += 1

doSave('bal_train')
doSave('eval')
doSave('unbal_train')