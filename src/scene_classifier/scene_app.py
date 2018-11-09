import argparse
import os.path as osp
import pickle
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter


class Scene():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = self.load_resnet(self.device)
        self.input_sz = 224
        self.i2t = transforms.ToTensor()
        self.total_num_frames = 30

    def load_resnet(self, device):
        resnet = models.resnet18(pretrained=True).to(device)
        # Remove the clf layer
        modules = list(resnet.children())[:-1]
        resnet = nn.Sequential(*modules)
        # Inference, no need to calc gradient
        for param in resnet.parameters():
            param.requires_grad = False

        return resnet

    def get_feats_per_frame(self, frame_path):
        img = Image.open(frame_path).resize((self.input_sz, self.input_sz), Image.ANTIALIAS)
        img_tensor = self.i2t(img).view(1, 3, self.input_sz, self.input_sz).to(self.device)
        output = self.resnet(img_tensor).cpu().view(-1)
        return output

    def predict(self, frames_path):
        svm = pickle.load(open('./data/scene_19_clf_svm_linear.pkl', 'rb'))
        pred = []
        for frame_path in frames_path:
            feats = self.get_feats_per_frame(frame_path)
            val = svm.predict(feats)
            pred.append(val)
        most_common, num_most_common = Counter(pred).most_common(1)[0]
        if num_most_common/self.total_num_frames > 0.9:
            return most_common
        else:
            return 'unk'
