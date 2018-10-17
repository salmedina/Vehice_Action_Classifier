import sys
import glob
import numpy as np


cfg_data_dir = '/mnt/sdd/tingyaoh/diva/DETRAC-Train-Data-Orientation/'
cfg_annot_fn = cfg_data_dir+'orientation_labels_b_16.csv'
cfg_fnlist = glob.glob(cfg_data_dir+'*.jpg')

### define splits by camera views
cfg_train_views = ['MVI_20011', 'MVI_20012', 'MVI_39761', 'MVI_39821', 'MVI_39851']
cfg_valid_views = ['MVI_40131', 'MVI_40141', 'MVI_40201']
cfg_test_views = ['MVI_40243', 'MVI_40244', 'MVI_40991']

def cfg_extract_data_split(lst, views):
    imgfnlist, degreelist, binlist = [],[],[]
    for l in lst:
        view_id = 'MVI_'+l.split(',')[0].split('_')[1]
        if view_id in views:
            ifn, degree, binn = l.split(',')
            imgfnlist.append(cfg_data_dir+ifn)
            degreelist.append(float(degree))
            binlist.append(int(np.floor(float(binn)/2)))
    return imgfnlist, degreelist, binlist

