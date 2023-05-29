import fnmatch
import os
import sys
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from torch.nn import functional as F
from torch.utils.data import Dataset

sys.path.append('./')

import random

from modelBuilder.torchvggish.vggish_input import wavfile_to_examples
from PIL import Image

from moco_SA.datalLoader.SA import SA, SAWithoutAug


def SA_(audio_path):
    ans=SAWithoutAug(audio_path)
    return ans

#加载整个的数据集
class AudioDatasetWithLabel(Dataset):
    def __init__(self,data_folder):
        self.data_folder=data_folder

        # self.augmentation = transforms.Compose([
        #     transforms.Resize([224, 224]),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.ToTensor(),#变成tensor
        #     transforms.Normalize(mean=[0.5],
        #                         std=[0.5])
        # ])
        self.wav_list=[]
        self.labels=['Dysphagia','Healthy']
        self.labels2id={'Dysphagia':0,'Healthy':1}

        for root, dirnames, filenames in os.walk(data_folder):
            for dirname in dirnames:
                # if dirname not in self.labels:
                #     self.labels2id[dirname]=self.labels.__len__()
                #     self.labels.append(dirname)
                for root_,dirnames_, filenames_ in os.walk(os.path.join(root,dirname)):
                    try:
                        for filename in fnmatch.filter(filenames_,"*.wav"):
                            self.wav_list.append((os.path.join(root,dirname,filename),self.labels2id[dirname]))
                    except UnicodeEncodeError:
                        pass
    def __getitem__(self, item):
        filename=self.wav_list[item][0]
        ans =wavfile_to_examples(filename,return_tensor=True)
        type=self.wav_list[item][1]
        return {'input':ans,'labels':torch.tensor(type), 'person': filename.split('/')[-1].split('_')[0], 'filename': filename}
    def __len__(self):
        return len(self.wav_list)

#针对单个类
class AudioDatasetWithLabel_single(Dataset):
    def __init__(self,data_folder,label):
        self.data_folder=data_folder

        # self.augmentation = transforms.Compose([
        #     transforms.Resize([244, 244]),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.ToTensor(),#变成tensor
        #     transforms.Normalize(mean=[0.5],
        #                         std=[0.5])
        # ])

        self.wav_list=[]
        self.labels=['Dysphagia','Healthy']
        self.labels2id={'Dysphagia':0,'Healthy':1}

        for root, dirnames, filenames in os.walk(data_folder):
            try:
                for filename in fnmatch.filter(filenames,"*.wav"):
                    # print('aaa')
                    self.wav_list.append((os.path.join(root,filename),self.labels2id[label]))
            except UnicodeEncodeError:
                pass
    def __getitem__(self, item):
        filename=self.wav_list[item][0]
        # print(filename.split('/')[-1].split('_')[0])
        ans =wavfile_to_examples(filename,return_tensor=True)
        # ans=SA_(filename)
        # ans=self.augmentation(ans)
        # type=[0,0]
        # type[self.wav_list[item][1]]=1
        type=self.wav_list[item][1]
        # return {'input':ans,'labels':torch.tensor(type), 'person': filename.split('/')[-1]}
        return {'input':ans,'labels':torch.tensor(type), 'person': filename.split('/')[-1].split('_')[0], 'filename': filename}
    def __len__(self):
        return len(self.wav_list)

# class AudioDatasetWithLabel_part(Dataset):
#     def __init__(self,data_folder):
#         self.data_folder=data_folder

#         self.augmentation = transforms.Compose([
#             transforms.Resize([96, 64]),
#             transforms.Grayscale(num_output_channels=1),
#             transforms.ToTensor(),#变成tensor
#             transforms.Normalize(mean=[0.5],
#                                 std=[0.5])
#         ])

#         self.wav_list=[]
#         self.labels=['Dysphagia','Healthy']
#         self.labels2id={'Dysphagia':0,'Healthy':1}

  
#         for root_,dirnames_, filenames_ in os.walk(os.path.join(data_folder,'Dysphagia')):
#             for filename in fnmatch.filter(filenames_,"*.wav"):
#                 # print(np.random.randint(0,1))
#                 # if((dirname=='Healthy') and np.random.randint(0,10)>8) or dirname=='Dysphagia':
#                 self.wav_list.append((os.path.join(data_folder,'Dysphagia',filename),self.labels2id['Dysphagia']))

#         # print(self.wav_list.__len__())
#         pathDir=[wav for wav in os.listdir(os.path.join(data_folder,'Healthy')) if wav.endswith('.wav')]
#         sample = random.sample(pathDir, self.wav_list.__len__())
#         for i in sample:
#             self.wav_list.append((os.path.join(data_folder,'Healthy',i),self.labels2id['Healthy']))
        
#         # print(self.wav_list.__len__())
#     def shuffle(self):
#         self.wav_list=[]

#         for root_,dirnames_, filenames_ in os.walk(os.path.join(self.data_folder,'Dysphagia')):
#             for filename in fnmatch.filter(filenames_,"*.wav"):
#                 self.wav_list.append((os.path.join(self.data_folder,'Dysphagia',filename),self.labels2id['Dysphagia']))
#         pathDir=[wav for wav in os.listdir(os.path.join(self.data_folder,'Healthy')) if wav.endswith('.wav')]
#         sample = random.sample(pathDir, self.wav_list.__len__())
#         for i in sample:
#             self.wav_list.append((os.path.join(self.data_folder,'Healthy',i),self.labels2id['Healthy']))


        
#     def __getitem__(self, item):
#         #返回 torch.tensor 才会正常的堆叠
#         filename=self.wav_list[item][0]
#         ans=SA_(filename)
#         ans=self.augmentation(ans)
#         type=[0.,0.]
#         type[self.wav_list[item][1]]=1.
#         return ans,torch.tensor(type)
#     def __len__(self):
#         return len(self.wav_list)



# train_dataset=AudioDatasetWithLabel_part('/home/server5/jwwang/Data/wav_data_label/train/')
# print(train_dataset.__len__())
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=4, 
#     num_workers=0, pin_memory=True, drop_last=True,shuffle=True)

# for i, (images, label) in enumerate(train_loader):
#     print(label)
