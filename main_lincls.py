#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pstats import Stats
import sys
from cmath import inf
from collections import defaultdict
from email.policy import default
from turtle import forward
from typing import List

sys.path.append(os.path.dirname(__file__))
sys.path.append("..")
sys.path.append("../../")
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
from pickletools import optimize
from xmlrpc.client import Boolean, boolean

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from datalLoader.wav_dataloader_label import (AudioDatasetWithLabel,
                                              AudioDatasetWithLabel_single)
from losses import LDAMLoss
from modelBuilder.model import NLLModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from modelBuilder.torchvggish import vggish_params

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

global initial_image
global grad_block
global fmap_block
global name
global fig
global axees
fig, axes = plt.subplots(figsize = (24, 12), nrows = 4, ncols = 4)

# initial_image = defaultdict()
# grad_block = defaultdict()
# fmap_block = defaultdict()

def main(parser):
    # 设置随机数种子
    args = parser.parse_args() # 解析命令行参数

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print('use device: {}'.format(args.cuda if torch.cuda.is_available() else -1))

    args.device = device

    setup_seed(args.seed)

    benchmarks = (
        ("Healthy", os.path.join(args.data_dir, 'test','Healthy')),
        ("Dysphagia", os.path.join(args.data_dir, 'test','Dysphagia')),
        ("Mix", os.path.join(args.data_dir, 'test')),
    )
    dev_benchmarks = (
        ("Healthy", os.path.join(args.data_dir, 'dev','Healthy')),
        ("Dysphagia", os.path.join(args.data_dir, 'dev','Dysphagia')),
        ("Mix", os.path.join(args.data_dir, 'dev')),
    )

    benchmarks_single=(
        ('Single_Test', '/home/server8/jwwang/Data/ExpData/single_test'),
    )

    if args.mode=="train":
        model=NLLModel(args)
        train(args,model,dev_benchmarks,benchmarks)
    elif args.mode=="single":
        single_test(args,benchmarks_single)
    else:
        test(args,benchmarks)
    

def backward_hook(module, grad_in, grad_out):
    global grad_block
    print(module)
    # print(grad_in[0].shape)
    print(grad_out[0].shape)
    grad_block=grad_out[0].cpu().detach().numpy()

def forward_hook(module, input, output):
    print("aaa")
    global fmap_block
    print(module)
    print(input[0].shape)
    print(output.shape)
    fmap_block=output.cpu().detach().numpy()

def cam_show_img(bid, out_dir='/home/server8/jwwang/Data/img/'):
    global initial_image
    global grad_block
    global fmap_block
    # print(fmap_block)
    global name
    global fig
    global axes

    fontdict_={'family' : 'SimHei', 'size'   : 16}
    supfont= { 'size'   : 20}
    supsupfont= {'size'   : 28}

    mapinfo={
        'False:Healthy': (3, 'Wrong Classified Healthy Sample'),
        'True:Healthy': (1, 'Right Classified Healthy Sample'),
        'False:Dysphagia': (4, 'Wrong Classified Dysphagia Sample'),
        'True:Dysphagia': (2, 'Right Classified Dysphagia Sample'),
    }

    bid = mapinfo[name[0]][0] -1
    name = [mapinfo[name[0]][1]]

    typename=['Inital Image', 'Selected Neutron 1', 'Selected Neutron 2', 'CAM Neutron']
    # 4s的时长, 来划分帧
    # _, H, W = batch

    def plot_inner(feature_index, row, column, initial_image_, fmap_block_):
        _,num, _, H, W = initial_image_.shape
        feature_map = fmap_block_[0]
        cam = np.zeros(feature_map.shape[1:], dtype = np.float32)
        cam = feature_map[feature_index, :, :]
        # print(cam)
        cv2.normalize(cam, cam, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cam = cv2.resize(cam, (H, W))
        norm = mpl.colors.Normalize(vmin=cam.min(), vmax=cam.max())
        mask = axes[row][column].imshow(cam, origin='lower', cmap='jet', aspect="auto", norm=norm,alpha=1 )

        initial_image_ = np.transpose(initial_image_, axes=[0, 1, 2, 4, 3])
        norm = mpl.colors.Normalize(vmin=initial_image_.min(), vmax=initial_image_.max())
        img = axes[row][column].imshow(initial_image_[0, 0, 0, :, :], cmap='RdYlGn_r', origin='lower', aspect="auto", norm=norm, alpha=0.5)
        axes[row][column].set_yticks([])
        if row != 3:
            axes[row][column].set_xticks([])
        else:
            plt.axes(axes[row][column])
            axes[row][column].xaxis.set_major_locator(ticker.LinearLocator(5))
            plt.xticks(plt.xticks()[0] , np.arange(0,5, 1))
            plt.xlabel("Time(s)", fontdict= fontdict_)
            plt.title(typename[column], y=-0.5, fontsize = supfont['size'])


    # 绘制第0列的图
    initial_image_ = np.transpose(initial_image, axes=[0, 1, 2, 4, 3])
    norm = mpl.colors.Normalize(vmin=initial_image_.min(), vmax=initial_image_.max())
    img = axes[bid][0].imshow(initial_image_[0, 0, 0, :, :], cmap='RdYlGn_r', origin='lower', aspect="auto", norm=norm, alpha=0.5)
    axes[bid][0].set_title(name[0], fontdict=fontdict_)
    locators= [round(i * ( vggish_params.NUM_MEL_BINS /30 )) for i in [0,6,12,18,24]]
    axes[bid][0].yaxis.set_major_locator(ticker.FixedLocator(locators))
    print(axes[bid][0].get_yticks())
    plt.axes(axes[bid][0])
    plt.yticks(axes[bid][0].get_yticks() , np.arange(0,5000, 1000)) 
    plt.ylabel("Frequency(Hz)", fontdict=fontdict_)
    if bid ==3:
        plt.axes(axes[bid][0])
        axes[bid][0].xaxis.set_major_locator(ticker.LinearLocator(5))
        plt.xticks(plt.xticks()[0] , np.arange(0,5, 1))
        plt.xlabel("Time(s)", fontdict= fontdict_)
    else:
        axes[bid][0].set_xticks([])


    plot_inner(4, bid, 2, initial_image, fmap_block)
    plot_inner(11, bid, 1, initial_image, fmap_block)



    _,num, _, H, W = initial_image.shape
    feature_map = fmap_block[0]
    grads = grad_block[0]
    cam = np.zeros(feature_map.shape[1:], dtype = np.float32)
    grads = grads.reshape([grads.shape[0], -1])
    weight = np.mean(grads, axis=1)
    for i, w in enumerate(weight):
        cam += w * feature_map[i, : , :]

    cv2.normalize(cam, cam, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cam = cv2.resize(cam, (H, W))
    norm = mpl.colors.Normalize(vmin=cam.min(), vmax=cam.max())
    mask = axes[bid][3].imshow(cam, origin='lower', cmap='jet', aspect="auto", norm=norm,alpha=1)

    initial_image_ = np.transpose(initial_image, axes=[0, 1, 2, 4, 3])
    norm = mpl.colors.Normalize(vmin=initial_image_.min(), vmax=initial_image_.max())
    if bid ==3:
        plt.axes(axes[bid][3])
        axes[bid][3].xaxis.set_major_locator(ticker.LinearLocator(5))
        plt.xticks(plt.xticks()[0] , np.arange(0,5, 1))
        plt.xlabel("Time(s)", fontdict= fontdict_)
        plt.title(typename[3], y=-0.5, fontsize = supfont['size'])
    else:
        axes[bid][3].set_xticks([])
    axes[bid][3].set_yticks([])

    plt.suptitle('Model Activation', fontsize = supsupfont['size'])
    plt.savefig(out_dir+ 'test_noAttention.jpg', format="jpg")


def single_test(args,benchmarks):
    model=NLLModel(args) 


    model.model.vggModel.features.register_full_backward_hook(backward_hook)
    model.model.vggModel.features.register_forward_hook(forward_hook)

    allmodel_pred=defaultdict(list)
    allmodel_pred_person=defaultdict(list)
    allgt=defaultdict(list)
    allgt_person=defaultdict(list)
    allperson=defaultdict(list)
    for index_,model_tmp in enumerate(args.model_list.split(',')):
        model_tmp=str(model_tmp).strip()
        print("Testing: model of {} from {}. *********".format(index_,model_tmp))
        state_dict=torch.load(model_tmp)['state_dict']
        # print(state_dict.keys())
        # state_dict.pop('models.0.fc.0.weight')
        model.load_state_dict(state_dict, strict=False)

        # print(benchmarks)
        for tag,feature in benchmarks:
            acc, pred, gt, acc_person, pred_person, gt_person, persons=evaluate(args,model,feature,tag=tag)
            print("pred: {}".format(pred_person))
            print("gt: {}".format(gt_person))

            allgt[tag]=gt
            allgt_person[tag]=gt_person
            for index, i  in enumerate(pred):
                if len(allmodel_pred[tag])<=index:
                    allmodel_pred[tag].append([])
                allmodel_pred[tag][index].append(i)
            for index, i_person in enumerate(pred_person):
                if len(allmodel_pred_person[tag]) <= index:
                    allmodel_pred_person[tag].append([])
                allmodel_pred_person[tag][index].append(i_person)
    
    print('*'*10)
    print("final one===>")
    for tag,feature in benchmarks:
        prediction=stats.mode(np.array(allmodel_pred[tag]), axis=len(np.array(allmodel_pred[tag]).shape)-1).mode.squeeze()
        correct_prediction = np.equal(prediction, allgt[tag])
        print('tag: {}----acc: {}'.format(tag,np.mean(correct_prediction)))

        prediction=stats.mode(np.array(allmodel_pred_person[tag]), axis=len(np.array(allmodel_pred_person[tag]).shape)-1).mode.squeeze()
        correct_prediction = np.equal(prediction, allgt_person[tag])
        print('person: tag: {}----acc: {}'.format(tag,np.mean(correct_prediction)))

    print('*'*10)


def test(args,benchmarks):
    model=NLLModel(args) 
    allmodel_pred=defaultdict(list)
    allmodel_pred_person=defaultdict(list)
    allgt=defaultdict(list)
    allgt_person=defaultdict(list)
    allpersons=defaultdict(list)
    for index_,model_tmp in enumerate(args.model_list.split(',')):
        model_tmp=str(model_tmp).strip()
        print("Testing: model of {} from {}. *********".format(index_,model_tmp))
        state_dict=torch.load(model_tmp)['state_dict']
        # state_dict.pop('models.0.fc.0.weight')
        model.load_state_dict(state_dict, strict=False)

        # print(benchmarks)
        for tag,feature in benchmarks:
            acc, pred, gt, acc_person, pred_person, gt_person, persons=evaluate(args,model,feature,tag=tag)

            allpersons[tag] = persons
            allgt[tag]=gt
            allgt_person[tag]=gt_person
            print('tag: {}----acc: {}'.format(tag,acc))
            print('preson: tag: {}----acc: {}'.format(tag,acc_person))
            for index, i  in enumerate(pred):
                if len(allmodel_pred[tag])<=index:
                    allmodel_pred[tag].append([])
                allmodel_pred[tag][index].append(i)
            for index, i_person in enumerate(pred_person):
                if len(allmodel_pred_person[tag]) <= index:
                    allmodel_pred_person[tag].append([])
                allmodel_pred_person[tag][index].append(i_person)
                
    print('*'*10)
    print("final one===>")
    for tag,feature in benchmarks:
        prediction=stats.mode(np.array(allmodel_pred[tag]), axis=len(np.array(allmodel_pred[tag]).shape)-1).mode.squeeze()
        correct_prediction = np.equal(prediction, allgt[tag])
        print('tag: {}----acc: {}'.format(tag,np.mean(correct_prediction)))

        # find sample
        if args.find_sample:
            if tag in ["Healthy","Dysphagia"]:
                print(tag)
                print('*' * 50)
                for name, pred, key, preds in zip(allpersons[tag], prediction, allgt[tag], allmodel_pred[tag]):
                    if pred == key:
                        print("{}: pred={}, key={}, {}\n\tpreds: {}".format(name, pred, key, pred==key, preds))
            if tag in ["Healthy","Dysphagia"]:
                print(tag)
                print('*' * 50)
                for name, pred, key, preds in zip(allpersons[tag], prediction, allgt[tag], allmodel_pred[tag]):
                    if pred != key:
                        print("{}: pred={}, key={}, {}\n\tpreds: {}".format(name, pred, key, pred==key, preds))


        prediction=stats.mode(np.array(allmodel_pred_person[tag]), axis=len(np.array(allmodel_pred_person[tag]).shape)-1).mode.squeeze()
        correct_prediction = np.equal(prediction, allgt_person[tag])
        print('person: tag: {}----acc: {}'.format(tag,np.mean(correct_prediction)))

    print('*'*10)



def train(args, model, dev_benchmarks, benchmarks):

    traindir = os.path.join(args.data_dir, args.train_name)

    train_dataset=AudioDatasetWithLabel(traindir)

    train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, drop_last=False,shuffle=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    # warmup_steps = int(total_steps * args.warmup_ratio)
    print('Total steps {}.'.format(total_steps))

    # scaler = GradScaler()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    num_steps = 0
    best_loss=inf
    best_acc=0
    for epoch in range(0, int(args.num_train_epochs)):
        print('Epoch {} ======='.format(epoch))
        # tqdm是结束完才跳进度条
        losses=[]
        for step, batch in enumerate(tqdm(train_dataloader)):

            model.train()
            # print(batch)
            #参数的warm_up 和整个训练过程有关
            if num_steps < int(args.alpha_warmup_ratio * total_steps):
                args.alpha_t = 0
            else:
                args.alpha_t = args.alpha

            # input, label
            batch = {key: value.to(args.device) for key, value in batch.items() if key!='person' and key!='filename'}
            # print(batch['input'].shape)
            outputs = model(**batch) # 模型的输出 (loss， logit)
            loss = outputs[0] / args.gradient_accumulation_steps # loss的累计操作(要除)
            loss.backward()
            losses.append(outputs[0].cpu().detach().numpy())

            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                optimizer.step()
                model.zero_grad() 
        
        dev_accs=defaultdict()
        for tag, features in dev_benchmarks:
            acc,_,_,_,_,_,_ = evaluate(args, model, features, tag=tag)
            print('[dev]---tag: {}----acc: {}'.format(tag,acc))
            dev_accs[tag]=acc

        accs=defaultdict()
        for tag, features in benchmarks:
            acc,_,_,_,_,_,_ = evaluate(args, model, features, tag=tag)
            print('[test]---tag: {}----acc: {}'.format(tag,acc))
            accs[tag]=acc

        average_acc = (dev_accs['Healthy'] + dev_accs['Dysphagia'])/2
        print('[dev]---average acc: {}'.format(average_acc))
        if average_acc > best_acc:
            print('[dev]---average acc improvement :{} --------'.format(average_acc-best_acc))
            state={'epoch':epoch,
            'state_dict': model.state_dict(),
           }
            torch.save(state,args.save_path)
            best_acc=average_acc
            print('save model to {}'.format(args.save_path))

        print('loss: {}--------'.format(np.mean(losses)))
        if np.mean(losses)< best_loss  or epoch < args.num_train_epochs * 0.4:
            best_loss=np.mean(losses)
        else:
            print("break for loss decrease ! ")
            break

def evaluate(args, model, features, tag):

    datadir=features
    if tag in ["Healthy","Dysphagia"]:
        dataset=AudioDatasetWithLabel_single(datadir,tag)
    else:
        dataset=AudioDatasetWithLabel(datadir)

    dataloader=torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, drop_last=False,shuffle=False)
    
    preds, keys = [], []
    persons=[]


    if tag !="Single_Test":
        with torch.no_grad():
            for batch in dataloader:
                model.eval()
                persons.extend(batch['person'])
                batch = {key: value.to(args.device) for key, value in batch.items() if key!='person' and key!='filename'}
                keys.extend(batch['labels'].detach().cpu().numpy().tolist())
                batch['labels'] = None
                output = model(**batch)
                logits=output[1]
                preds.extend(torch.argmax(logits, dim=-1).tolist())

    else:
        for bid, batch in enumerate(dataloader):
            model.eval()
            persons.extend(batch['person'])
            global name
            name = batch['person']

            batch = {key: value.to(args.device) for key, value in batch.items() if key!='person'and key!='filename' }
            keys.extend(batch['labels'].detach().cpu().numpy().tolist())
            batch['labels'] = None
            output = model(**batch)
            # print(batch['input'].shape)
            global initial_image 
            initial_image=batch['input'].cpu().detach()
            # print(initial_image.shape)
            logits=output[1]
            preds.extend(torch.argmax(logits, dim=-1).tolist())
            
            if tag=="Single_Test":
                # print(logits[0][keys[0]])
                logits[0][keys[0]].backward()
            cam_show_img(bid) # 绘制图形

    preds = np.array(preds, dtype=np.int32)
    keys = np.array(keys, dtype=np.int32)

    #group by person
    groupbyperson_preds=defaultdict(list)
    groupbyperson_keys=defaultdict(list)
    for pred, key, person in zip(preds, keys, persons):
        groupbyperson_keys[person].append(key)
        groupbyperson_preds[person].append(pred)
    
    person_preds=[]
    person_keys=[]
    person=groupbyperson_keys.keys()
    for preds_, keys_ in zip(groupbyperson_preds.values(), groupbyperson_keys.values()):
        person_preds.append(stats.mode(np.array(preds_), axis=0).mode[0])
        person_keys.append(stats.mode(np.array(keys_), axis=0).mode[0])


    correct_prediction = np.equal(preds, keys)
    correct_prediction_person = np.equal(person_preds, person_keys)

    return np.mean(correct_prediction), preds, keys, np.mean(correct_prediction_person), person_preds, person_keys, persons



if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--cuda', default=0, type=int, metavar='N',
                        help='cuda')
    parser.add_argument("--learning_rate", default=6e-5, type=float)
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6) # 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="NLL-IE-IC")
    parser.add_argument("--n_model", type=int, default=1)
    parser.add_argument("--baise", type=int, default=1)

    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)


    parser.add_argument('--is_pretrained',  type=str2bool, default=False,
                        help='wthether to pretrain')            
    parser.add_argument('--pretrained_path',  type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('--data_dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument("--save_path",type=str,default='/home/server5/jwwang/Data/outcome',help="as named")
    parser.add_argument("--model_name_or_path", default="the structure of model", type=str)
    
    parser.add_argument("--train_name", type=str, default="down-train")

    parser.add_argument('--mode',  type=str, default="train",
                        help='running mode')
    parser.add_argument('--num_model', default=5, type=int, metavar='N',
                        help='number of trained model to do test')
    parser.add_argument('--model_list', type=str, default='', help='<Required> when test')


    parser.add_argument('--use_attention',  type=str2bool,
                        help='wthether to attention layer in vggish')  
    parser.add_argument('--k', default=5, type=int,
                    help='attention layer for each class.')

    parser.add_argument('--use_resize', default='true', type=str2bool,
                    help='')
    
    parser.add_argument('--find_sample',  type=str2bool, default=False, 
                        help='wthether to attention layer in vggish')  

    main(parser)
