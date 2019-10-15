import torchvision
from torch import nn
from torch.nn import functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes, use_pretrained=True, **kwargs):
        super(ResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=use_pretrained)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.classifier = nn.Linear(resnet.fc.in_features, num_classes)
        self.feat_dim = resnet.fc.in_features # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        return y

class OrientationResNet18(nn.Module):
    def __init__(self, use_pretrained=True, **kwargs):
        super(OrientationResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=use_pretrained)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.feat_dim = resnet.fc.in_features # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        return y


class MultitaskResNet18(nn.Module):
    def __init__(self, orientation_sz=360, direction_sz=4, use_pretrained=True, **kwargs):
        super(MultitaskResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=use_pretrained)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.orientation = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, orientation_sz))
        self.direction = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, direction_sz))
        self.feat_dim = resnet.fc.in_features # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y_orientation, y_direction = self.orientation(f), self.direction(f)
        return y_orientation, y_direction