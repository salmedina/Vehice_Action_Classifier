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

train_ids_list = [['000000','000001','000002','000003','000004','000005','000006'],
                    ['000101'],
                    ['000200','000201','000202','000203','000204','000205','000206'],
                    ['010000','010001','010002','010003','010004'],
                    ['010100','010101','010102','010103','010104','010105','010106','010107','010108','010109','010110','010111','010112','010113'],
                    ['010200','010201','010202','010203','010204','010205','010206'],
                    ['040000','040001','040002','040003','040004'],
                    ['040100','040101','040102','040103'],
                    ['050000'],
                    ['050100'],
                    ['050200','050201','050202','050203'],
                    ['050300']]

test_ids_list = [['000007', '000008'],
                  ['000102'],
                  ['000207'],
                  ['010005'],
                  ['010114','010115','010116'],
                  ['010207','010208'],
                  ['040005'],
                  ['040104'],
                  ['050000'],
                  ['050101'],
                  ['050204'],
                  ['050301']]


def get_num_scene_samples(scene_ids_list, total_samples):
    return int(total_samples / len(scene_ids_list))

def get_all_scene_frame_paths(scene_ids_list, scene_id, frames_path):
    all_scene_videos = []
    for video_id in scene_ids_list[scene_id]:
        video_frame_paths = glob(osp.join(frames_path, 'VIRAT_S_%s*' % (video_id), '*.jpg'))
        all_scene_videos += video_frame_paths
    return all_scene_videos

def sample_scene_frame_paths(scene_ids_list, frames_path, total_samples):
    num_samples_per_scene = get_num_scene_samples(scene_ids_list, total_samples)

    sampled_scenes_frames = []
    for i in range(len(scene_ids_list)):
        scene_frames = get_all_scene_frame_paths(scene_ids_list, i, frames_path)
        sampled_frames = [scene_frames[idx] for idx in
                          np.random.choice(len(scene_frames), num_samples_per_scene, replace=False)]
        sampled_scenes_frames.append(sampled_frames)

    return sampled_scenes_frames

def load_resnet(device):
    resnet = models.resnet18(pretrained=True).to(device)
    # Remove the clf layer
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    # Inference, no need to calc gradient
    for param in resnet.parameters():
        param.requires_grad = False

    return resnet

def get_frame_rel_path(frame_path):
    return osp.join(osp.basename(osp.dirname(frame_path)), osp.basename(frame_path))

def main(frames_path, save_path, num_samples, ids_list):
    input_sz = 224
    sampled_frames = sample_scene_frame_paths(ids_list, frames_path, num_samples)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = load_resnet(device)
    i2t = transforms.ToTensor()
    samples = []
    for sid, scene_frames in enumerate(sampled_frames):
        print('Extracting frames for scene:', sid)
        for frame_path in scene_frames:
            img = Image.open(frame_path).resize((input_sz, input_sz), Image.ANTIALIAS)
            img_tensor = i2t(img).view(1, 3, input_sz, input_sz).to(device)
            output = resnet(img_tensor).cpu().view(-1)
            samples.append((get_frame_rel_path(frame_path), sid, output.data.numpy()))

    print('Total samples:', len(samples))
    with open(save_path, 'wb') as save_file:
        pickle.dump(samples, save_file)
    print('Saved feats to', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet extractor')
    parser.add_argument('-i', dest='inputDir', help='Directory where the video frames are located')
    parser.add_argument('-s', dest='numSamples', type=int, help='Number of total samples')
    parser.add_argument('-o', dest='output', help='Path to the output pickle file')
    args = parser.parse_args()

    ids_list = train_ids_list
    main(args.inputDir, args.output, args.numSamples, ids_list)