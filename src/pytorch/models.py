import sys

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class ResNet50(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        return y
        
class ResNet18(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet18, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(resnet18.children())[:-2])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        return y
