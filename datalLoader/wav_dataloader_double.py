import os
import sys
from ast import Try
from turtle import forward, shape

# from sqlalchemy import true
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(__file__,'../../')))
sys.path.append(os.path.abspath(os.path.join(__file__,'../../../')))

# print(sys.path)

import argparse
import fnmatch
import os
# import librosa
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modelBuilder.torchvggish.vggish_input import wavfile_to_examples
from torch.nn import functional as F
from torch.utils.data import Dataset


class AudioDatasetDouble(Dataset):
    def __init__(self,args,data_dir) -> None:
        self.args=args
        self.data_dir=data_dir
        self.wav_list=[]

        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                self.wav_list.append(os.path.join(data_dir,file))
    
    def random_crop(self,data, crop_size=128):
        #mel图是 时间*频率(mel) 时间划分子sample

        while data.shape[0]<crop_size:
            # print(data.shape)
            data=np.concatenate([data,data],axis=0)
            # print(data.shape)

        start = int(random.random() * (data.shape[0] - crop_size))
        return data[start : (start + crop_size), :]


    def random_multiply(self,data):
        new_data = data.copy()
        return new_data * (0.9 + random.random() / 5.)
    def random_mask(self,data, rate_start=0.1, rate_seq=0.2):
        new_data = data.copy()
        mean = new_data.mean()
        prev_zero = False
        for i in range(new_data.shape[0]):
            if random.random() < rate_start or (
                prev_zero and random.random() < rate_seq
            ):
                prev_zero = True
                new_data[i, :] = mean
            else:
                prev_zero = False

        return new_data

    def __len__(self) -> int:
        return len(self.wav_list)
    def __getitem__(self, index: int):
        filename=self.wav_list[index]



        try:
            # 最后以0.96s的时长进行组帧 [时长/0.96, 96, 64]
            inputs=wavfile_to_examples(filename,return_tensor=False)
        except:
            os.remove(filename)
            return (0),(0)

        if self.args.augment:
            inputs=self.random_mask(inputs)
        
        x1=self.random_crop(inputs,crop_size=self.args.max_len)
        x2=self.random_crop(inputs,crop_size=self.args.max_len)

        if self.args.augment:
            x1=self.random_multiply(x1)
            x2=self.random_multiply(x2)

        x1 = torch.tensor(x1, dtype=torch.float).to(self.args.device)
        x2 = torch.tensor(x2, dtype=torch.float).to(self.args.device)
        return {'image_q':x1, 'image_k':x2}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='dataset testing')

    args = parser.parse_args()

    args.data_dir='/home/server5/jwwang/Data/CoswareDataset/train'
    args.augment=True
    args.max_len=4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device=device

    dataset=AudioDatasetDouble(args,args.data_dir)

    for x1,x2 in dataset:
        print(x1.shape)
        print(x2.shape)


    
