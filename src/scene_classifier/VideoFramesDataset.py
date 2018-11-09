import os.path as osp
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VideoFramesDataset(Dataset):

    def __init__(self, video_dir, num_frames, size):
        '''
        Loads the first num_frames frames of the video
        :param video_dir: dir path for the video clip to be processed in batch
        :param num_frames: number of video frames to be processed
        '''

        def path_num_val(path):
            return int(osp.splitext(osp.basename(path))[0])

        frames_path_list = glob(osp.join(video_dir, '*.jpg'))
        frames_path_list.sort(key=path_num_val)
        self.path_list = frames_path_list[:num_frames]
        self.size = size
        self.length = len(self.path_list)
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.path_list[index]).resize((self.size, self.size), Image.ANTIALIAS)
        img_as_tensor = self.transforms(img)
        return img_as_tensor

    def __len__(self):
        return self.length