import csv
import os.path as osp
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from VideoFramesDataset import VideoFramesDataset


def load_resnet(device):
    resnet = models.resnet18(pretrained=True).to(device)
    # Remove the clf layer
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    # Inference, no need to calc gradient
    for param in resnet.parameters():
        param.requires_grad = False

    return resnet

def get_resnet_feats(img_path, resnet, device):
    img = Image.open(img_path).resize((224, 224), Image.ANTIALIAS)
    img_tensor = transforms.ToTensor()(img).view(1, 3, 224, 224).to(device)
    output = resnet(img_tensor).cpu().view(-1)
    return output.data.numpy()


def plot_cm(cm, class_names, figsize, fontsize):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='rocket')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title('SVM (Linear)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':

    data_path = '/home/zal/Data/VIRAT/Frames/first_frames'
    device = torch.device('cpu')
    resnet = load_resnet(device)
    svm_model = pickle.load(open('/home/zal/Devel/Vehice_Action_Classifier/output/scene_19_clf_svm_linear.pkl', 'rb'))

    # Annotations is a csv with videoclip subdir name and label
    anno_path = osp.join(data_path, 'annotations.csv')
    anno_data = []
    with open(anno_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        anno_data = list(reader)
    data_x, data_y = zip(*anno_data)
    data_y = [int(y) for y in data_y]

    cm = np.zeros((19, 19), dtype=np.int)
    for video_dir, y in zip(data_x, data_y):
        videoclip_path = osp.join(data_path, video_dir)
        print(videoclip_path, end=', ')
        resnet_feats = []
        video_dataset = VideoFramesDataset(osp.join(data_path, video_dir), 30, 224)
        video_dataloader = DataLoader(video_dataset, batch_size=30, shuffle=False, num_workers=16)
        frames_tensor = next(iter(video_dataloader))
        resnet_feats = resnet(frames_tensor).reshape((30,512)).data.numpy()
        frame_prediction = svm_model.predict(resnet_feats)
        vid_prediction = np.argmax(np.bincount(frame_prediction))
        print(y, vid_prediction)
        if y != vid_prediction:
            print(frame_prediction)
        cm[y, vid_prediction] += 1

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(np.sum(np.diag(cm)))
    print(np.sum(cm))
    print('Accuracy:',accuracy)
    plot_cm(cm.astype(np.int), class_names=[str(i) for i in range(19)], figsize=(10,7), fontsize=14)
