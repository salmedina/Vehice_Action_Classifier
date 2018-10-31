import glob

person_trk_dir = '/home/tingyaoh/research/diva/tracking/tracklets/offline/person/'
car_trk_dir = '/home/tingyaoh/research/diva/tracking/tracklets/offline/car/'
#car_trk_dir = '/mnt/sdd/tingyaoh/diva/tracklets/offline/car/'
#person_trk_dir = '/mnt/sdd/tingyaoh/diva/tracklets/offline/person/'

diva_util_path = './'
#diva_annot_path = '/mnt/sdd/tingyaoh/diva/annotation/'
diva_annot_path = '/data/MM1/tingyaoh/diva_v1/annotation/v1-drop4/'
train_annot_path = diva_annot_path+'train-kpf/'
valid_annot_path = diva_annot_path+'validate-kpf/'

#img_path = '/home/tingyaoh/research/diva/tracking/imgs/'

train_vid_list = [l.split('/')[-1].split('.')[0] for l in glob.glob(train_annot_path+'*.activities.yml')]
valid_vid_list = [l.split('/')[-1].split('.')[0] for l in glob.glob(valid_annot_path+'*.activities.yml')]

VEHICLE_ACT_NAMES = ['vehicle_turning_right','vehicle_turning_left','vehicle_u_turn']
