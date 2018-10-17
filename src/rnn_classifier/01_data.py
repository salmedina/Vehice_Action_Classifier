import sys
import os
from myconfig import *
sys.path.append(diva_util_path)
from diva_util import *
from collections import defaultdict
import numpy as np
from action_proposal_util import Vehicle
import pickle

# tracklet: list of (t, bbox)
def get_time_bbox_dict(tracklet):
    bbox_dict = defaultdict()
    for t, box in tracklet:
        bbox_dict[t] = box
    return bbox_dict

def xywh2xxyy(box):
    x,y,w,h = box
    return x,y,x+w,y+h

"""
vehicle feature functions
    tracklet: list of (t, bbox)
"""
def velocity_estimation(tracklet, dist=3):
    vlst = []
    for i in range(len(tracklet)):
        t, box = tracklet[i]
        if i+dist>=len(tracklet): break
        v = v_estimate(t, box, tracklet[i+dist][0], tracklet[i+dist][1])
        v = np.linalg.norm(v)
        vlst.append(v)
    if len(vlst)==0: return 0
    return np.mean(vlst)
        
def v_estimate(t1, box1, t2, box2):
    x1, y1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    x2, y2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    v = [(x2-x1), (y2-y1)]
    return v

def angle_velocity(tracklet, dist=5):
    wlst = []
    for i in range(len(tracklet)):
        t, box = tracklet[i]
        if i+2*dist>=len(tracklet): break
        w = w_estimate(t, box, tracklet[i+dist][0], tracklet[i+dist][1], tracklet[i+2*dist][0], tracklet[i+2*dist][1])
        wlst.append(w)
    if len(wlst)==0: return 0
    return np.mean(wlst)
    
def w_estimate(t1,box1,t2,box2,t3,box3):
    x1, y1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    x2, y2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    x3, y3 = (box3[0]+box3[2])/2, (box3[1]+box3[3])/2
    v1,v2 = [x2-x1,y2-y1], [x3-x2,y3-y2]
    cos = ((x2-x1)*(x3-x2)+(y2-y1)*(y3-y2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
    t_1, t_2 = (t1+t2)/2, (t2+t3)/2
    if np.isnan(cos): return 0
    cos = max(min(1,cos),-1)
    if cos<0: return (3.1416-np.arccos(cos))/(t_2-t_1)
    return np.arccos(cos)/(t_2-t_1)

seglen = 60

cmd = 'mkdir -p data/'
os.system(cmd)

for vid in train_vid_list:
    featlst = []
    lablst = []
    
    actlst = parse_diva_act_yaml(train_annot_path+vid+'.activities.yml')
    geomlst = parse_diva_geom_yaml(train_annot_path+vid+'.geom.yml')
    geom_id_dict = get_geom_id_list(geomlst)
    typedict = parse_diva_type_yaml(train_annot_path+vid+'.types.yml')

    ### extract all ground truth tracklet
    
    act_tubes = []
    act_idxs = []
    for act in actlst:
        if 'meta' in act.keys(): continue
        if act['act2'] not in VEHICLE_ACT_NAMES: continue
        span, bb_lst = get_act_tubelet(act, geom_id_dict)
        actlab = VEHICLE_ACT_NAMES.index(act['act2'])+1
        act_tubes.append((actlab, span, bb_lst))
        act_idxs.append(actlab)
    matched = [0]*len(act_tubes)

    ### extract detection result
    trackdict = read_mot_as_defaultdict(car_trk_dir+vid+'.txt')
    for k in trackdict.keys():
        start = min([t for t, box in trackdict[k]])
        end = max([t for t, box in trackdict[k]])
        if end-start<10: continue
        
        labs = np.zeros((end-start+1,))
        bbox_dict = get_time_bbox_dict(trackdict[k])
        for j,act_tube in enumerate(act_tubes):
            actlab, span, bb_lst = act_tube
            s2,e2 = span
            if tiou(start,end,s2,e2)>0: 
                    iou_values = []
                    for tt in range(max(start,s2),min(end,e2)+1):
                        try:
                            iou_values.append(iou(xywh2xxyy(bbox_dict[tt]), bb_lst[tt-s2]))
                        except KeyError:
                            continue
                            print('tracker missing')
                    if np.mean(iou_values)>0.8: 
                        #print('matched!',k, max(start,s2),min(end,e2))
                        s, e = max(start,s2),min(end,e2)
                        labs[s-start:e-start+1] = actlab
                        matched[j] = 1

        #if np.sum(labs)!=0: print(labs)
        car = Vehicle(k, trackdict[k])
        vmean = np.mean([np.linalg.norm(car.vs[i,:]) for i in range(len(car.vs))])
        trk_lab = 1 if np.sum(labs)>0 else 0
        #print('average v:',vmean, trk_lab)
        if vmean<1 and trk_lab==1: print(('average v:',vmean, trk_lab))
        if vmean>=1:
            featlst.append(car.vs)
            lablst.append(labs)

    print(vid, matched,act_idxs)
    pickle.dump((featlst,lablst), open('data/'+vid+'.pkl','wb'))
