import torch.nn as nn
import torch
from torch.nn import functional as F
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()
    def forward(self, predict,label):
        beta=1-torch.mean(label)
        weight=1-beta+(2*beta-1)*label
        loss=F.binary_cross_entropy(predict,label,weight,size_average=False)
        return loss

