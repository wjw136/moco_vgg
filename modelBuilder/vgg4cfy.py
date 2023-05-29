# 绝对路径导入
# 相对路径导入

# 将模块路径加入到系统路径中
import os
import sys
from cmath import log
from turtle import forward

sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvggish.vggish import VGGish, vggish_params
import math


class vggForcfy(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.args=args
        # print(args.use_resize)
        self.vggModel = VGGish(args.model_name_or_path,use_attention=args.use_attention,k=args.k,num_class=args.num_class,pretrained=False, use_resize=args.use_resize)

        if args.is_pretrained:
            #加载预训练模型参数 因为有修改过模型的结构 需要特殊加载 
            # print("aaa:{}".format(args.pretrained_path))
            print(">>>>load pretrain model from {}".format(args.pretrained_path))
            pretrain_model=torch.load(args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict'] if 'state_dict' in pretrain_model.keys() else pretrain_model
            model_dict=self.vggModel.state_dict()
            pretrain_dict={k:v for k,v in state_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.vggModel.load_state_dict(model_dict, strict=False)

        # self.fc= nn.Sequential(
        #     nn.Linear(128* math.ceil(3/vggish_params.NUM_FRAMES * 100),args.num_class),
        #     nn.Softmax() #要加激活函数
        # )
        self.fc= nn.Sequential(
            nn.Linear(128 ,args.num_class),
            nn.Softmax() #要加激活函数
        )

        self.loss=nn.CrossEntropyLoss()
    
    def forward(self,inputs,labels):

        bs, num_samples,_,_,_ =inputs.shape
        inputs=inputs.view(bs*num_samples,1,inputs.size(3),inputs.size(4))


        outputs=self.vggModel(inputs)

        outputs=outputs.view(bs,num_samples*outputs.size(1)) #平铺的操作


        outputs=self.fc(outputs)

        logics=outputs  

        if labels!=None:
            loss=self.loss(logics,labels)
        else:
            loss=torch.tensor(0)
        
        return (loss,logics)
