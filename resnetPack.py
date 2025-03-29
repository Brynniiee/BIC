from functools import reduce
import torch
from torch import nn, autograd
import torchvision.models as models
from torch.nn import functional as F
import utils

import torch.nn as nn
import math

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(f"in layer {i}, alpha = {self.alpha.item()}, beta = {self.beta.item()}")

def resnet50(num_classes):
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    return model

def resnet152(num_classes):
    model = models.resnet152(pretrained=False, num_classes=num_classes)
    return model

def resnet34(num_classes): 
    model = models.resnet34(pretrained=False, num_classes=num_classes)
    return model