import os.path as osp
from glob import glob

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VideoFramesDataset(Dataset):

    def __init__(self, video_dir, num_frames, transforms=None):
        '''
        Loads the first num_frames frames of the video
        :param video_dir: dir path for the video clip to be processed in batch
        :param num_frames: number of video frames to be processed
        '''

        def path_num_val(path):
            return osp.splitext(osp.basename(path))[0]

        frames_path_list = glob(osp.join(video_dir, '*.jpg'))
        frames_path_list.sort(key=path_num_val)
        self.path_list = frames_path_list[:num_frames]
        self.length = len(self.path_list)
        if transforms is None:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.path_list[index])
        img_as_tensor = self.transforms(img)
        return (img_as_tensor)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    transformations = transforms.Compose([transforms.ToTensor()])
    video_frames_dir = '/Users/zal/CMU/Projects/DIVA/Data/Frames/VIRAT_S_040104_09_001475'
    videoframes_dataset = VideoFramesDataset(video_frames_dir, num_frames=30, transforms=transformations)
    videoframes_dataset_loader = DataLoader(dataset=videoframes_dataset, batch_size=10, shuffle=False)

    for images in videoframes_dataset_loader:
        print(len(images))
        print(images)
