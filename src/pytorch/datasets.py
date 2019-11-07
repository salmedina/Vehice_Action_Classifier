import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class DegOrientationDataset(Dataset):

    def __init__(self, imgfnlist, degreelist, binlist, transform=None):
        self.imgfnlist, self.degreelist, self.binlist = imgfnlist, degreelist, binlist
        self.transform = transform

    def __getitem__(self, idx):
        imgfn, angle, binn = self.imgfnlist[idx], self.degreelist[idx], self.binlist[idx]
        img = read_image(imgfn)
        if self.transform is not None:
            img = self.transform(img)

        return img, int(angle), int(binn)

    def __len__(self):
        return len(self.imgfnlist)


class MevaOrientationDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        data = [l.strip().split(',') for l in open(csv_path).readlines()]
        self.imgfnlist = list()
        self.labellist = list()
        self.transform = transform

        self.imgfnlist, self.labellist = zip(*[(osp.join(image_dir, fn), int(label)) for fn, label in data])
        # for imagefn, label in [(osp.join(image_dir, fn), int(label)) for fn, label in data]:
        #     if osp.exists(imagefn):
        #         self.imgfnlist.append(imagefn)
        #         self.labellist.append(label)

    def __getitem__(self, idx):
        imgfn, binn = self.imgfnlist[idx], self.labellist[idx]
        img = read_image(imgfn)
        if self.transform is not None:
            img = self.transform(img)

        return img, int(binn)

    def __len__(self):
        return len(self.imgfnlist)