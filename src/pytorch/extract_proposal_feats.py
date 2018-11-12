import os
import os.path as osp
import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from models import ResNet18

i2t = transforms.ToTensor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Torch device:', device)

def load_resnet(model_path, num_bins):
    resnet = ResNet18(num_classes=num_bins)
    resnet.load_state_dict(torch.load(model_path))
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules).to(device)
    for param in resnet.parameters():
        param.requires_grad = False
    return resnet

def extract_frames(video_path, skip_frame=8):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    frames = []
    while success:
      if count%8 == 0:
        frames.append(image)
      success,image = vidcap.read()
      count += 1
    return frames

def extract_feats(resnet, video_frames, device):
    feats = np.ndarray((len(video_frames), 7, 7, 512))
    for idx, frame in enumerate(video_frames):
        frame_img = Image.fromarray(frame).resize((224, 224), Image.ANTIALIAS)
        img_tensor = i2t(frame_img).view(1, 3, 224, 224).to(device)
        output = resnet(img_tensor).transpose(1,2).transpose(2,3)
        feats[idx, :, :, :] = output.cpu()
    return feats

def safe_mkdirs(path)
    if not osp.exists(path):
        os.makedirs(path)

def main(source_dir, model_path, num_bins, save_dir, video_name):
    resnet = load_resnet(model_path, num_bins)

    for proposal in os.listdir(source_dir):
        safe_mkdirs(osp.join(save_dir, proposal))
        for videoclip_dir in os.listdir(osp.join(source_dir, proposal)):
            print('Extracting feats of', videoclip_dir)
            video_path = osp.join(source_dir, proposal, videoclip_dir, video_name)
            frames = extract_frames(video_path, 8)
            feats = extract_feats(resnet, frames, device)
            feats_save_path = osp.join(save_dir, proposal, '%s.npz' % videoclip_dir)
            np.savez_compressed(feats_save_path, feat=feats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposals', type=str, help='Directory with proposals')
    parser.add_argument('--numbins', type=int, help='Number of orientation bins of model')
    parser.add_argument('--model', type='str', help='Path for orientation model')
    parser.add_argument('--savedir', type=str, help='Save directory for orientation feats')
    parser.add_argument('--videoname', type=str, default='video.mp4', help='Name of video for each proposal')
    args = parser.parse_args()

    source_dir = '/media/zal/Alfheim/Data/VIRAT/proposals/'
    save_dir = '/media/zal/Alfheim/Data/VIRAT/orientation/'
    model_path = '/home/zal/Devel/Vehice_Action_Classifier/src/pytorch/models/resnet18_pretrained_bs_128_aug_rot_acc_70v0.pt'

    main(source_dir, model_path, 16, save_dir, 'video.mp4')
    # main(args.proposals, args.model, args.num_bins, args.savedir, video_name)