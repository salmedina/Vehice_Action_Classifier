import os
import sys
import numpy as np
import json
import cv2
import glob
import pickle
from collections import defaultdict

ACT_NAMES = ['Person_Person_Interaction', 'Object_Transfer', 'Closing_Trunk', 'SetDown', 'specialized_talking_phone', 'Riding', 'vehicle_moving', 'activity_gesturing', 'Unloading', 'Loading', 'vehicle_stopping', 'Open_Trunk', 'Opening', 'specialized_miscellaneous', 'activity_running', 'vehicle_turning_left', 'PickUp_Person_Vehicle', 'Pull', 'activity_walking', 'activity_crouching', 'DropOff_Person_Vehicle', 'Drop', 'Exiting', 'Transport_HeavyCarry', 'Push', 'Closing', 'vehicle_starting', 'activity_sitting', 'vehicle_u_turn', 'activity_standing', 'activity_carrying', 'Misc', 'Talking', 'specialized_using_tool', 'Entering', 'vehicle_turning_right', 'specialized_texting_phone', 'Interacts','PickUp','specialized_throwing','specialized_umbrella']
ACT_NAMES_V1 = ['vehicle_turning_right','vehicle_turning_left','Closing_Trunk','Closing','Opening','Exiting','Entering','Transport_HeavyCarry','Unloading','Loading','Open_Trunk','vehicle_u_turn']
ACT_NAMES_V1b = ['vehicle_turning_right','vehicle_turning_left','Closing_Trunk','Closing','Opening','Exiting','Entering','Transport_HeavyCarry','Unloading','Loading','Open_Trunk','vehicle_u_turn', 'Pull', 'Interacts', 'Riding', 'Talking', 'activity_carrying', 'specialized_talking_phone', 'specialized_texting_phone']
#ACT_NAMES = ['Talking','Closing_Trunk','Closing','Entering','Exiting','Interacts','Loading','Open_Trunk','Opening','Person_Person_Interaction','Pull','Push','SetDown','Transport_HeavyCarry','Unloading','Existing', 'Entering', 'activity_walking', 'activity_running', 'activity_carrying', 'vehicle_moving','specialized_using_tool','activity_standing','activity_gesturing','vehicle_moving','vehicle_starting','vehicle_stopping','vehicle_turning_left','vehicle_turning_right','specialized_miscellaneous','specialized_talking_phone','specialized_texting_phone']
KWS = ['meta','ts0','poly0','act2','id0','id1','id2','timespan','tsr0','src','actors','occlusion','g0','truth','keyframe']

#########################################
##             Visualization           ##
#########################################

def vis_detection(img, bboxes):
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    cv2.imshow('det', img)
    cv2.waitKey(-1)

def vis_event(img_fn_lst, event_tracklet,detdir=None):
    print 'act id:',event_tracklet['id2']
    start = event_tracklet['start']
    print [len(e) for e in event_tracklet['tls']]
    eventl = len(event_tracklet['tls'][0])
    print start, eventl, 
    actor_num = len(event_tracklet['tls'])

    detdir = '/data/MM1/tingyaoh/diva_v1/object_detection/fullres_resnet101_thres0.05_pnms500_json/'
    for i in range(start,start+eventl):
        img = cv2.imread(img_fn_lst[i])
        """
        for j in range(actor_num):
            bbox = event_tracklet['tls'][j][i-start]
            print bbox
            x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        """

        ### plot detection
        det_bboxes = []
        if detdir is not None:
            detfn = detdir+vid+'_F_'+str(i).zfill(8)+'.json'
            jobj = json.loads(file(detfn).readline())
            for det in jobj:
                x1,y1,x2,y2 = det['bbox']
                x1,y1,x2,y2 = int(x1),int(y1),int(x1+x2),int(y1+y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                

        bbox = event_tracklet['ctl'][i-start]
        print bbox
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        img = cv2.resize(img,None,fx=0.5,fy=0.5)
        cv2.imshow('event', img)
        cv2.waitKey(-1)

#############################################
#      diva YAML parsers (using json)       #
#############################################

def get_scene_id(video_id):
    return video_id.split('_')[2][:4]

def keyword2str(l, kwlst, single=False, v2action=False):
    for kw in kwlst:
        if not v2action: l = l.replace(kw,'\"'+kw+'\"')
        else: l.replace(kw,'\"'+kw[:-1]+'\"')
        if kw in l and single: break
    return l

def parse_diva_region_yaml(fn):
    s = '['
    for l in file(fn):
        l = l[2:].strip()
        l = keyword2str(l,KWS)
        if ',' in l:
            idx = l.rindex(',')
            l = l[:idx]+l[idx+1:]
        s+=l+','
    s = s[:-1]
    s+=']'
    return json.loads(s)

def parse_diva_geom_yaml(fn):
    geomlst = []
    for l in file(fn):
        if 'meta' in l: continue
        lst = l.split()
        geom = {}
        #geom['id0'] = int(lst[5][:-1])
        #geom['id1'] = int(lst[3][:-1])
        #geom['ts0'] = int(lst[7][:-1])
        #geom['bbox'] = int(lst[11]), int(lst[12]), int(lst[13]), int(lst[14])
        geom['id0'] = int(lst[7][:-1])
        geom['id1'] = int(lst[5][:-1])
        geom['ts0'] = int(lst[9][:-1])
        geom['bbox'] = int(lst[13]), int(lst[14]), int(lst[15]), int(lst[16])
        geomlst.append(geom)
    return geomlst

def parse_diva_type_yaml(fn):
    type_dict = {}
    for l in file(fn):
        if 'id1' in l:
            lst = l.split()
            type_dict[int(lst[5])] = lst[9][:-1]
    return type_dict

def parse_diva_act_yaml(fn):
    s = '['
    for l in file(fn):
        #l = l[2:].strip()
        if 'meta' in l: l=l[2:].strip()
        else: l = l.strip()[9:-2]
        l = keyword2str(l,KWS)
        if 'meta' not in l:
            lst = l.split()
            lst[3] = lst[3][:-1]
            if lst[6]!=',': lst[3]+=','
            l = ' '.join(lst[:2]+[lst[3]]+lst[6:])
        if 'meta' not in l: l = keyword2str(l,ACT_NAMES,single=True)
        if l[-4]==',':
            idx = l.rindex(',')
            l = l[:idx]+l[idx+1:]
        s+=l+','
    s = s[:-1]
    s+=']'
    return json.loads(s)

###############################################
#       Action Detection util functions       #
###############################################
"""
given object id, extract bounding boxes in temporal range (start,end) from geomlst for this oid
"""
def get_bboxes(oid, geomlst_oid, start, end):
    oid_start = geomlst_oid[0]['ts0']
    bb_lst = []
    for t in range(start, end+1):
        try:
            bbox = geomlst_oid[t-oid_start]['bbox']
        except IndexError:
            print 'object absence', oid, t
        try:
            bb_lst.append(bbox)
        except UnboundLocalError:
            return []
    return bb_lst

"""
get an id specific geomlst from total goemlst
defaultdict required
"""
def get_geom_id_list(geomlst):
    geom_id_dict = defaultdict(list)
    for geom in geomlst:
        geom_id_dict[geom['id1']].append(geom)
    ## sort
    for oid in geom_id_dict.keys():
        geom_id_dict[oid] = sorted(geom_id_dict[oid], key=lambda geom:geom['ts0'])
    return geom_id_dict


###############################################
#            Tracking util functions          #
###############################################

"""
region_js: json object of region yaml file (VIRAT_S_XXXXXX.regions.yml)
type_js: jsoin object of type yaml file (VIRAT_S_XXXXXX.type.yml)
"""
def gen_mot_annotation(region_js, type_js, obj_type=['Vehicle','Person']):
    region_js = sorted(region_js[1:], key = lambda i:i['ts0'])
    for region in region_js:
        if type_js[region['id1']] not in obj_type: continue
        x1,y1,x2,y2 = get_bbox(region)
        out = str(region['ts0'])+','+str(region['id1'])+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+',-1,-1,-1,-1'
        print out
    
def get_bbox(region):
    x1,y1 = region['poly0'][0]
    x2,y2 = region['poly0'][2]
    return x1,y1,x2,y2


def get_obj_tracklets(geomlst,type_dict):
    tracklet_dict = {}
    geomlst = sorted(geomlst, key = lambda i:i['ts0'])
    for id1 in type_dict.keys():
        geom_sub_lst = [geom for geom in geomlst if geom['id1']==id1]
        #if len(regionlst)==0: continue
        bb_lst = [geom['bbox'] for geom in geom_sub_lst]

        tracklet_dict[id1] = {}
        tracklet_dict[id1]['bboxes'] = bb_lst
        tracklet_dict[id1]['start'] = geom_sub_lst[0]['ts0']
    return tracklet_dict

def get_act_tracklet(act_js, tracklet_dict):
    event_tracklet_lst = []
    for act in act_js:
        if 'meta' in act.keys(): continue
        if act['act2'] not in ACT_NAMES_V1: continue
        bb_lsts = []
        e_tl = {}
        #print act['id2']
        for actor in act['actors']:
            start,end = actor['timespan'][0]['tsr0']
            #print start,end
            id1 = actor['id1']
            try:
                objstart = tracklet_dict[id1]['start']
                start,end = start-objstart, end-objstart
                bb_lst = tracklet_dict[id1]['bboxes'][start:end+1]
                if len(bb_lst)>0:
                    bb_lsts.append(bb_lst)
            except KeyError:
                print 'warning: unseen obj', id1

        if len(bb_lsts)==0:
            print 'warning: action with no actor', act['id2']
            continue
        
        e_tl['act2'] = act['act2']
        e_tl['id2'] = act['id2']
        e_tl['tls'] = bb_lsts
        e_tl['start'] = act['timespan'][0]['tsr0'][0]
        combined_bblst = []
        eventl = np.amax([len(bb_lst) for bb_lst in bb_lsts])
        for i in range(eventl):
            x1 = np.amin([ bb_lst[-1][0] if i>=len(bb_lst) else bb_lst[i][0] for bb_lst in bb_lsts])
            y1 = np.amin([ bb_lst[-1][1] if i>=len(bb_lst) else bb_lst[i][1] for bb_lst in bb_lsts])
            x2 = np.amax([ bb_lst[-1][2] if i>=len(bb_lst) else bb_lst[i][2] for bb_lst in bb_lsts])
            y2 = np.amax([ bb_lst[-1][3] if i>=len(bb_lst) else bb_lst[i][3] for bb_lst in bb_lsts])
            combined_bblst.append([x1,y1,x2,y2])
        e_tl['ctl'] = combined_bblst
        if len(combined_bblst)==0:
            print 'warning: zero length act',act['id2']
            continue
        event_tracklet_lst.append(e_tl)
    return event_tracklet_lst

def iou(bb_test,bb_gt):
    ### bb_*: [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w*h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
            + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

def combine_bbs(bb1,bb2):
    x11,y11,x12,y12 = bb1
    x21,y21,x22,y22 = bb2
    x1,y1,x2,y2 = np.maximum(x11,x21),np.maximum(y11,y21),np.maximum(x12,x22),np.maximum(y12,y22)
    return [x1,y1,x2,y2]

def tiou(event_tracklet1, event_tracklet2, combine=False):
    start1,start2 = event_tracklet1['start'], event_tracklet2['start']
    end1,end2 = start1+len(event_tracklet1['ctl']), start2+len(event_tracklet2['ctl'])
    if start1>=end2 or start2>=end1: return 0,0,{}
    start, end = np.maximum(start1,start2), np.minimum(end1,end2)
    l = end-start
    idx_start1,idx_start2 = start-start1, start-start2
    idx_end1,idx_end2 = idx_start1+l,idx_start2+l
    #print start,end,idx_start1,idx_end1,idx_start2,idx_end2
    ioulst = []
    combined_bblst = []
    for i1,i2 in zip(range(idx_start1,idx_end1),range(idx_start2,idx_end2)):
        bb1,bb2 = event_tracklet1['ctl'][i1], event_tracklet2['ctl'][i2]
        if bb1 is None: print event_tracklet1['ctl']
        if bb2 is None: print event_tracklet2['ctl']
        ioulst.append(iou(bb1,bb2))
        if combine: combined_bblst.append(combine_bbs(bb1,bb2))
    to = float(end-start)/((end1-start1)+(end2-start2)-(end-start))
    combined_tl = {}
    if combine:
        combined_tl['start'] = start
        combined_tl['ctl'] = combined_bblst
    return to, np.mean(ioulst), combined_tl

"""
Generate ground truth file in MOT Challenge format
output as output str list
"""
def gen_mot_gt(geomlst, type_dict, otype='Person',actors=None):
    geomlst = sorted(geomlst, key=lambda geom:geom['ts0'])
    outlst = []
    for geom in geomlst:
        x1,y1,x2,y2 = geom['bbox']
        if type_dict[geom['id1']]==otype and (actors is None or geom['id1'] in actors):
        #if type_dict[geom['id1']]==otype and actors is None:
            outl = str(geom['ts0']+1)+','+str(geom['id1'])+','+str(x1)+','+str(y1)+','+str(x2-x1)+','+str(y2-y1)+',-1,-1,-1,-1'
            outlst.append(outl)
    return outlst

   
"""
for each tracklet, store it as a list of (timestamp, bbox)
"""
def read_mot_as_defaultdict(mot_output_fn):
    trackdict = defaultdict(list)
    for l in open(mot_output_fn):
        lst = [int(i) for i in l.split(',')]
        t = lst[0]-1
        box = lst[2:6]
        trackdict[lst[1]].append((t,box))
    return trackdict


def get_act_tubelet(act, geom_id_dict):
    bb_lsts, spans = [], []
    for actor in act['actors']:
        oid = actor['id1']
        span = actor['timespan'][0]['tsr0']
        #print span
        bb_lst = get_bboxes(oid, geom_id_dict[oid], span[0], span[1])
        if len(bb_lst)>0:
            bb_lsts.append(bb_lst)
            spans.append(span)
    if len(bb_lsts)==1:
        bb_lst, span = bb_lsts[0], spans[0]
    else:
        bb_lst = [ combine_box(box1,box2) for box1, box2 in zip(bb_lsts[0], bb_lsts[1])]
        span = [np.maximum(spans[0][0], spans[1][0]), np.minimum(spans[0][1], spans[1][1])]
    return span, bb_lst

def tiou(s1,e1,s2,e2):
    return max(0,float(min(e1,e2)-max(s1,s2))/(max(e1,e2)-min(s1,s2)))

if __name__=='__main__':
    
    #rdir = '/mnt/gpu6/sdc/tingyaoh/diva/annotation/'
    #tldir = '/mnt/gpu6/sdc/tingyaoh/diva/ground_truth_tracklet/'
    rdir = '/data/MM1/tingyaoh/diva_v1/annotation/'
    tldir = '/data/MM1/tingyaoh/diva_v1/ground_truth_tracklet/'
    #annot_dir_lst = glob.glob(rdir+'*/VIRAT_S*')
    annot_dir_lst = ['/data/MM1/tingyaoh/diva_v1/annotation/0000/VIRAT_S_000000']
    for annot_dir in annot_dir_lst:
        print annot_dir
        fn = glob.glob(annot_dir+'/*.geom*')[0]
        geomlst = parse_diva_geom_yaml(fn)
        fn = glob.glob(annot_dir+'/*.type*')[0]
        type_dict = parse_diva_type_yaml(fn)
        fn = glob.glob(annot_dir+'/*.activities*')[0]
        act_js = parse_diva_act_yaml(fn)
        tracklet_dict = get_obj_tracklets(geomlst,type_dict)
        event_tracklet_lst = get_act_tracklet(act_js, tracklet_dict)
        vid = fn.split('/')[-2]
        os.system('mkdir -p '+tldir+vid)
        pickle.dump(event_tracklet_lst,open(tldir+vid+'/gt_tracklet.pkl','wb'))
        print len(event_tracklet_lst)
        gen_mot_gtfile(geomlst,file('tmp','w'))

    """
    img_dir = 'imgs/VIRAT_S_000000'
    img_fn_lst = glob.glob(img_dir+'/*.jpg')
    img_fn_lst = sorted(img_fn_lst,key=lambda fn:int(fn.split('/')[-1].split('.')[0]))
    vis_event(img_fn_lst, event_tracklet_lst[40])
    """
 
