import argparse
import os.path as osp
import pickle
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import ffmpeg


class Scene():
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_path = args.resnet_path
        self.resnet = self.load_resnet(self.device)
        self.input_sz = 224
        self.i2t = transforms.ToTensor()
        self.total_num_frames = 15
        self.svm = pickle.load(open(args.model, 'rb'))
        self.video_dir = args.video_dir
        self.video_lst_files = args.video_lst_file
        self.output_dir = args.out_dir
        self.tmp_path = '/code/tmp_frames/'

    def load_resnet(self, device):
        resnet = models.resnet18(pretrained=False).to(device)
        resnet.load_state_dict(torch.load(self.resnet_path))
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

    def valid(self, path):
        if path.endswith('.jpg') or path.endswith('.png'):
            return True
        else:
            return False

    def predict(self, frames_path):
        pred = []
        for frame_path in frames_path:
            feats = self.get_feats_per_frame(frame_path)
            # val = self.svm.predict(feats.reshape(1, 512))[0]
            probs = abs(self.svm.predict_proba(feats.reshape(1, 512)))
            # print(probs.max())
            if probs.max() > 0.9:
                val = 1
            else:
                val = 0
            pred.append(val)
        most_common, num_most_common = Counter(pred).most_common(1)[0]
        print(pred)
        return most_common
        # if num_most_common/self.total_num_frames > 0.9:
        #     return most_common
        # else:
        #     return 

    def parse_video(self, video_path):
        frames_path = self.tmp_path
        os.system("ffmpeg -i {0} -loglevel panic -vf fps=fps=1 -vframes {2} {1}/output%d.png".format(video_path, frames_path, self.total_num_frames))

    def run(self):
        r = []
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with open(self.video_lst_files, "r") as f:
            for line in f:
                sc.parse_video(self.video_dir + "/" + str(line.strip()))
                path = self.tmp_path
                paths = os.listdir(path)
                res = []
                for file_ in paths:
                    res.append(path+str(file_))
                r = sc.predict(res)
                with open(self.output_dir+"/"+str(line.strip())+".camera", "w") as g:
                    g.write(str(r))
        return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Classification')
    parser.add_argument('--video_dir', dest='video_dir', type=str, default='', help='the root directory path of videos')
    parser.add_argument('--video_lst_file', dest='video_lst_file', type=str, default='', help='the path of video list, in this file each line is the relative path of the video to the video_dir. That is, video_file_path = os.path.join(video_dir, ${line}) (default: None) Note ${line} may contain "/" (default: None)')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='', help='the root directory of outputs: the camera id of each video is stored in the corresponding file "${out_dir}/${line}.camera. (default: None)')
    parser.add_argument('--m', dest='model', type=str, default='app/scene_12_clf_svm_linear.pkl', help='Model')
    parser.add_argument('--resnet_model', dest='resnet_path', type=str, default='app/resnet18-5c106cde.pth', help='Resnet18 model')
    args = parser.parse_args()
    # call the scene class and run it
    sc = Scene(args)
    print(sc.run())
