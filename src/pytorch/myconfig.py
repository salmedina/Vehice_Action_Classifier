import re
import numpy as np
from os.path import join

cfg_data_dir = '/home/zal/Data/DETRAC-Train-Data-Orientation'
cfg_anno_path = join(cfg_data_dir, 'orientation_labels_b_4.csv')

### define splits by camera views
cfg_night_views = ['MVI_39851', 'MVI_39861', 'MVI_40962', 'MVI_40963', 'MVI_40981', 'MVI_40991', 'MVI_40992']
cfg_train_views = ['MVI_20012', 'MVI_20032', 'MVI_20033', 'MVI_20034', 'MVI_20051', 'MVI_20052', 'MVI_20061', 'MVI_20062', 'MVI_20063', 'MVI_20064', 'MVI_20065', 'MVI_39761', 'MVI_39771', 'MVI_39781', 'MVI_39801', 'MVI_39811', 'MVI_39821', 'MVI_39931', 'MVI_40131', 'MVI_40141', 'MVI_40152', 'MVI_40161', 'MVI_40172', 'MVI_40181', 'MVI_40191', 'MVI_40192', 'MVI_40201', 'MVI_40204', 'MVI_40211', 'MVI_40212', 'MVI_40213', 'MVI_40241', 'MVI_40243', 'MVI_40244', 'MVI_40732', 'MVI_40751', 'MVI_40752', 'MVI_40871', 'MVI_40992', 'MVI_41063', 'MVI_63521', 'MVI_63525', 'MVI_63544', 'MVI_63552', 'MVI_63553', 'MVI_63554', 'MVI_63561', 'MVI_63562', 'MVI_63563']
cfg_valid_views = ['MVI_20011', 'MVI_20035', 'MVI_40162', 'MVI_40171', 'MVI_41073']

def cfg_extract_data_split(anno_list, views):
    imgfnlist, degreelist, binlist = [],[],[]
    for line in anno_list:
        view_id = re.search(r'MVI_\d+', line).group(0)
        if view_id in views:
            filename, degree, binn = line.split(',')
            imgfnlist.append(join(cfg_data_dir, filename))
            degreelist.append(float(degree))
            binlist.append(int(np.floor(float(binn)/2)))
    return imgfnlist, degreelist, binlist

def cfg_split_data(anno_list, split_views):
    data_list = [line.split(',') for line in anno_list]
    split_views_set = set(split_views)
    split_data = [(join(cfg_data_dir, filename), float(orientation), float(binn))
                  for filename, orientation, binn in data_list if re.search(r'MVI_\d+', filename).group(0) in split_views_set]
    imgpathlist, degreelist, binlist = zip(*split_data)

    return imgpathlist, degreelist, binlist
