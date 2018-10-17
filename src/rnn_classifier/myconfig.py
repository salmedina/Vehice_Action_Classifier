import glob

car_trk_dir = '/mnt/sdd/tingyaoh/diva/tracklets/offline/car/'
person_trk_dir = '/mnt/sdd/tingyaoh/diva/tracklets/offline/person/'

diva_util_path = './'
diva_annot_path = '/mnt/sdd/tingyaoh/diva/annotation/'
train_annot_path = diva_annot_path+'train-kpf/'
valid_annot_path = diva_annot_path+'validate-kpf/'

#img_path = '/home/tingyaoh/research/diva/tracking/imgs/'

train_vid_list = [l.split('/')[-1].split('.')[0] for l in glob.glob(train_annot_path+'*.activities.yml')]
valid_vid_list = [l.split('/')[-1].split('.')[0] for l in glob.glob(valid_annot_path+'*.activities.yml')]

VEHICLE_ACT_NAMES = ['vehicle_turning_right','vehicle_turning_left','vehicle_u_turn']
