"""
moco based on vgg model
"""
import os
import sys
from turtle import forward

# from sqlalchemy import true
sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from modelBuilder.moco.builder import MoCo
from torch import tensor
from torch.optim import Adam
from torchvggish.vggish import VGGish


class vgg4moco(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args=args

        #写死了encoder的输出
        self.encoder_q=VGGish(args.model_name_or_path, use_attention=args.use_attention,k=args.k,num_class=args.num_class,pretrained=False, use_resize=args.use_resize)
        self.encoder_k=VGGish(args.model_name_or_path, use_attention=args.use_attention,k=args.k,num_class=args.num_class,pretrained=False, use_resize=args.use_resize)
        if args.is_pretrained:
            print(">>>>load pretrain model from {}".format(args.pretrained_path))
            pretrained_model=torch.load(args.pretrained_path)
            state_dict = pretrained_model['state_dict'] if 'state_dict' in pretrained_model.keys() else pretrained_model
            model_dict=self.encoder_q.state_dict()
            pretrain_dict={k:v for k,v in state_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.encoder_q.load_state_dict(model_dict, strict=False)
            self.encoder_k.load_state_dict(model_dict, strict=False)

        #Moco是分布式的
        # dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
        self.model=MoCo(args=args,encoder_q=self.encoder_q,
                encoder_k=self.encoder_k,m=args.moco_m,
                T=args.moco_t,K=args.moco_k)
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2), eps=args.eps
        )
    
    def forward(self, args,image_q,image_k, isTrain=False):
        

        if isTrain:
            # print("bbbb")
            self.model.train()

            logics,labels=self.model(image_q,image_k)

            loss=self.loss(logics,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.cpu().detach().numpy()
        else:
            self.model.eval()

            logics,labels=self.model(image_q,image_k)

            logics=torch.argmax(logics,-1).cpu().detach().numpy()

            labels=labels.cpu().detach().numpy()

            return np.mean(np.equal(logics,labels))


    

        
