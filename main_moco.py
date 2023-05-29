#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
from cmath import inf

sys.path.append('./')

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from xmlrpc.client import boolean

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from datalLoader import wav_dataloader_double
from matplotlib.pyplot import step
from modelBuilder.vgg4moco import vgg4moco
from sklearn import impute
from tqdm import tqdm


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main(parser):
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print('use device: {}'.format(args.cuda if torch.cuda.is_available() else -1))

    args.device=device

    setup_seed(args.seed)

    main_worker(args)

    

def main_worker( args):
    print('use_attention=> {}'.format(args.use_attention))
    trainset=wav_dataloader_double.AudioDatasetDouble(args,os.path.join(args.data_dir,'train'))
    testset=wav_dataloader_double.AudioDatasetDouble(args,os.path.join(args.data_dir,'test'))
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, 
        num_workers=0, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size,
        num_workers=0, pin_memory=False, drop_last=True)

    model=vgg4moco(args)

    model.to(args.device)
    print("model prepare finishs")
    
    
    best_acc=0
    best_loss=inf
    for epoch in range(args.epochs):
        losses=[]
        acces=[]
        for index,inputs in enumerate(tqdm(train_loader)):
            # print(index)
            # if index % 100 ==0:
            #     print('train: {}/{}******'.format(index,len(train_loader)))
            losses.append(model(args=args,isTrain=True,**inputs))

        for index,inputs in enumerate(tqdm(test_loader)):
            # if index % 100 ==0:
            #     print('test: {}/{}******'.format(index,len(test_loader)))
            acc=model(args=args,isTrain=False,**inputs)
            acces.append(acc)
        
        print('epoch:{}-------acc:{}---------loss:{}'.format(epoch,
               np.mean(acces), np.mean(losses) ))
        
        if np.mean(acces)>=best_acc:
            best_acc=np.mean(acces)
            state={
                'epoch': epoch,
                'state_dict': model.encoder_q.state_dict(),
                'acc': np.mean(acces)
            }
            torch.save(state,args.save_path)
            print('save model dict to {}'.format(args.save_path))
        
        if np.mean(losses)<best_loss:
            best_loss=np.mean(losses)
        else:
            print("break for the loss increasing")
            break

# def save_checkpoint(args,state, epoch,acc):
#     torch.save(state, os.path.join(args.save_dir,'epoch:{}_acc:{}_checkpoint.pth.tar'.format(epoch,acc)))




if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--cuda', default=0, type=int, metavar='N',
                        help='cuda')

    parser.add_argument( '--learning_rate', default=0.03, type=float,
                         help='initial learning rate')
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6) # 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')

    parser.add_argument('--seed', default=10, type=int,
                        help='as named.')


    parser.add_argument('--data_dir', type=str,
                    help='as named.')
    parser.add_argument('--model_name_or_path', type=str,
                    help='as named.')
    parser.add_argument('--is_pretrained',  type=str2bool,
                        help='wthether to pretrain')  
    parser.add_argument('--pretrained_path',  type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('--save_path', type=str,
                    help='as named.')
    parser.add_argument('--augment', type=boolean,
                    help='whether to add the MASK&MUTIPLY operation.')


    # # moco specific configs:
    # parser.add_argument('--moco-dim', default=128, type=int,
    #                     help='feature dimension (default: 128)')
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    parser.add_argument('--use_attention',  type=str2bool,
                        help='wthether to attention layer in vggish')  
    parser.add_argument('--k', default=5, type=int,
                    help='attention layer for each class.')
    parser.add_argument('--num_class', default=2, type=int,
                    help=' num of class in final outcome.')


    parser.add_argument('--max_len', default=4, type=int,
                    help=' num of class in final outcome.')
    
    parser.add_argument('--use_resize', default='true', type=str2bool,
                    help='')


    main(parser)

    
