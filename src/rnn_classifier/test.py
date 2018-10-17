import sys
from myconfig import *
sys.path.append(diva_util_path)
from diva_util import *

seglen = 40

for vid in train_vid_list:
    
    actlst = parse_diva_act_yaml(train_annot_path+vid+'.activities.yml')
    geomlst = parse_diva_geom_yaml(train_annot_path+vid+'.geom.yml')
    geom_id_dict = get_geom_id_list(geomlst)
    typedict = parse_diva_type_yaml(train_annot_path+vid+'.types.yml')

    ### extract all ground truth tracklet
    for act in actlst:
        if 'meta' in act.keys(): continue
        if act['act2'] not in VEHICLE_ACT_NAMES: continue
        span, bb_lst = get_act_tubelet(act, geom_id_dict)
        print(len(bb_lst))

    ### extract detection result
    trackdict = read_mot_as_defaultdict(car_trk_dir+vid+'.txt')
    for k in trackdict.keys():
        start = min([t for t, box in trackdict[k]])
        end = max([t for t, box in trackdict[k]])

        for t in range(start, end, seglen/2):
            s,e = t, min(t+seglen, end)
            
