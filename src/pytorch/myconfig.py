import glob
from os.path import join

cfg_data_dir = '/home/zal/Data/DETRAC/DETRAC-Train-Data-Orientation-Aug/'
cfg_annot_fn =  join(cfg_data_dir, 'orientation_labels_b_16.csv')
cfg_fnlist = glob.glob(join(cfg_data_dir,'*.jpg'))

### define splits by camera views
cfg_train_views = ['MVI_20011','MVI_20061','MVI_39801','MVI_40152','MVI_40201','MVI_40732','MVI_40992','MVI_63554',
'MVI_20012','MVI_20062','MVI_39811','MVI_40161','MVI_40204','MVI_40751','MVI_41063','MVI_63561',
'MVI_20032','MVI_20063','MVI_39821','MVI_40162','MVI_40211','MVI_40752','MVI_41073','MVI_63562',
'MVI_20033','MVI_20064','MVI_40171','MVI_40212','MVI_40871','MVI_63521','MVI_63563','MVI_20034',
'MVI_20065','MVI_40172','MVI_40213','MVI_63525','MVI_20035','MVI_39761','MVI_39931','MVI_40181',
'MVI_40241','MVI_63544','MVI_20051','MVI_39771','MVI_40131','MVI_40191','MVI_40243','MVI_63552',
'MVI_20052','MVI_39781','MVI_40141','MVI_40192','MVI_40244','MVI_63553']
cfg_valid_views = ['MVI_40131', 'MVI_40141', 'MVI_40201']
cfg_test_views = ['MVI_40243', 'MVI_40244']

def cfg_extract_data_split(lst, views):
    imgfnlist, degreelist, binlist = [],[],[]
    for l in lst:
        view_id = 'MVI_'+l.split(',')[0].split('_')[1]
        if view_id in views:
            ifn, degree, binn = l.split(',')
            imgfnlist.append(join(cfg_data_dir, ifn))
            degreelist.append(float(degree))
            binlist.append(int(binn))
    return imgfnlist, degreelist, binlist

