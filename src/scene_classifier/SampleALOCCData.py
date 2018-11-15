import numpy as np
import csv
import os.path as osp
import pickle
from PIL import Image
from glob import glob
from random import shuffle
from joblib import Parallel, delayed
import multiprocessing as mp

def read_img_as_ndarray(img_path):
    try:
        img = Image.open(img_path).resize((64, 64)).convert('L')
    except OSError:
        print('OSError:', img_path)
        return None
    return np.asarray(img, dtype=np.uint8)

# Load the directories and the labels of the dirs
data_path = '/home/zal/Data/VIRAT/Frames/imgs'
anno_path = '/home/zal/Data/VIRAT/Frames/first_frames/annotations.csv'
anno_data = []
with open(anno_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    anno_data = list(reader)
data_x, data_y = zip(*anno_data)
data_y = np.array([int(y) for y in data_y])

# Load, downsample and grayscale images
samples_per_class = 10000
all_imgs = []
labels = []
for label in range(19):
    print('Loading images for label', label)
    label_idx = np.where(data_y == label)[0]
    label_clips = [data_x[i] for i in label_idx]
    label_frames = []
    for clip in label_clips:
        label_frames += list(glob(osp.join(data_path, clip, '*.jpg')))

    shuffle(label_frames)
    label_imgs = Parallel(n_jobs=mp.cpu_count())(delayed(read_img_as_ndarray)(p) for p in label_frames[:samples_per_class])

    all_imgs += label_imgs
    labels += [label]*len(label_imgs)

#Save to file
if len(labels) == len(all_imgs):
    data_save_path = '/home/zal/Devel/Vehice_Action_Classifier/output/alocc_data.npz'
    np.savez_compressed(open(data_save_path, 'wb'), images=np.stack(all_imgs), labels=np.array(labels))
    print('END! Successfully saved data to', data_save_path)
else:
    print('ERROR! Number of labels and images are not the same:', len(labels), len(all_imgs))
