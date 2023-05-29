import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelBuilder.vgg4cfy import vggForcfy


def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)


# 多个model的KL散度.
class NLLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.device = args.device
        self.loss_fnt = nn.CrossEntropyLoss()

        self.model=vggForcfy(args)
        self.model.to(self.device)

    def forward(self, input,  labels=None):
        
        output = self.model(
            inputs=input.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )
        output = tuple([o.to(0) for o in output])

        #NLL loss
        # model_output = outputs[-1]
        # if labels is not None:
        #     loss = output[0]
        #     logit = output[1]
        #     prob = F.softmax(logits, dim=-1)
        #     # reg_loss = k1_div(avg_prob, prob) 没有多个模型, 就没有正则损失 TODO: 加个正则损失


        #     loss = sum([output[0] for output in outputs]) / num_models
        #     logits = [output[1] for output in outputs]
        #     probs = [F.softmax(logit, dim=-1) for logit in logits]
        #     avg_prob = torch.stack(probs, dim=0).mean(0)
        #     reg_loss = sum([kl_div(avg_prob, prob) for prob in probs]) / num_models
        #     loss = loss + self.args.alpha_t * reg_loss.mean()
        #     model_output = (loss,) + model_output[1:] + (reg_loss,) # 元组相加是拼接
        return output
