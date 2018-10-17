import sys
import numpy as np
from collections import defaultdict
from sklearn.cluster import AffinityPropagation

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

def bbox_interpolation(box1,box2,t1,t2,t):
    x11,y11,x12,y12 = box1
    x21,y21,x22,y22 = box2
    x1 = int(float(x11*(t2-t)+x21*(t-t1))/(t2-t1))
    x2 = int(float(x12*(t2-t)+x22*(t-t1))/(t2-t1))
    y1 = int(float(y11*(t2-t)+y21*(t-t1))/(t2-t1))
    y2 = int(float(y12*(t2-t)+y22*(t-t1))/(t2-t1))
    return [x1,y1,x2,y2]

def oned_segmentation(ts):
    tlst = []
    seglst = []
    for t in ts:
        if len(tlst)==0 or np.abs(tlst[-1]-t)<30: tlst.append(t)
        else:
            start, end = min(tlst), max(tlst)
            tlst = []
            if end-start>30: seglst.append((start,end))
    if len(tlst)>0:
        start, end = min(tlst), max(tlst)
        tlst = []
        if end-start>30: seglst.append((start,end))
    return seglst


class Vehicle(object):

    """
    st_boxes: [(t,bbox)]
    """
    def __init__(self, oid, st_boxes):
        #self.st_boxes = sorted(st_boxes, lambda stbox:stbox[0])

        self.bboxes, self.timestamps = [],[]
        for stbox in st_boxes:
            t, box = stbox[0], stbox[1]
            x1,y1,w,h = box
            box = [x1,y1,x1+w,y1+h]
            if len(self.bboxes)>0 and t > self.timestamps[-1]+1:
                for tt in range(self.timestamps[-1]+1,t):
                    box2 = bbox_interpolation(self.bboxes[-1], box, self.timestamps[-1], t, tt)
                    self.bboxes.append(box2)
                    self.timestamps.append(tt)
            self.bboxes.append(box)
            self.timestamps.append(t)
        self.start,self.end = min(self.timestamps), max(self.timestamps)
        self.oid = oid

        self.motion_analysis()
        self.interaction_dict = defaultdict(list)

    def motion_analysis(self):
        wsize = 10
        centers = np.array([[x1+w/2,y1+h/2] for x1,y1,w,h in self.bboxes], dtype='float32')
        self.vs = []
        for i in range(len(self.bboxes)):
            s = min(max(0, i-wsize/2),len(self.bboxes)-wsize)
            v = np.mean(np.array([(centers[t+wsize/2]-centers[t]) for t in range(s,s+wsize/2)]), axis=0)
            self.vs.append(v)
        self.vs = np.array(self.vs)
                
    def interaction_analysis(self, person):
        p_start, p_end = person.start, person.end
        tiou = float(min(self.end,p_end)-max(self.start,p_start))/(max(p_end, self.end)-min(self.start, p_start))
        if tiou<=0: return
        start, end = max(self.start,p_start), min(self.end,p_end)
        for t in range(start,end+1):
            box1,box2 = self.get_box_from_t(t), person.get_box_from_t(t)
            iou_value = iou(box1,box2)
            if iou_value>0.001:
                self.interaction_dict[t].append(person.oid)
        
    def get_box_from_t(self,t):
        t = t-self.start
        if t>=0: return self.bboxes[t]

    def action_proposal_static(self, person_dict):
        ts = [t for t in range(self.start, self.end+1) if len(self.interaction_dict[t])>0]
        seglst = oned_segmentation(ts)
        proposals = []
        for start,end in seglst:
            start, end = max(start,self.start), min(end,self.end)

            car_boxes = [self.get_box_from_t(t) for t in range(start,end+1)]
            person_boxes = []
            for t in range(start,end+1):
                for pid in self.interaction_dict[t]:
                    person_boxes.append(person_dict[pid].get_box_from_t(t))
            car_boxes, person_boxes = np.array(car_boxes), np.array(person_boxes)
            x1 = np.minimum(np.min(car_boxes[:,0]),np.min(person_boxes[:,0]))
            y1 = np.minimum(np.min(car_boxes[:,1]),np.min(person_boxes[:,1]))
            x2 = np.maximum(np.max(car_boxes[:,2]),np.max(person_boxes[:,2]))
            y2 = np.maximum(np.max(car_boxes[:,3]),np.max(person_boxes[:,3]))
            proposals.append((start,end,x1,y1,x2,y2))
        return proposals


class Person(object):
    def __init__(self, oid, st_boxes):
        self.bboxes, self.timestamps = [],[]
        for stbox in st_boxes:
            t, box = stbox[0], stbox[1]
            x1,y1,w,h = box
            box = [x1,y1,x1+w,y1+h]
            if len(self.bboxes)>0 and t > self.timestamps[-1]+1:
                for tt in range(self.timestamps[-1]+1,t):
                    box2 = bbox_interpolation(self.bboxes[-1], box, self.timestamps[-1], t, tt)
                    self.bboxes.append(box2)
                    self.timestamps.append(tt)
            self.bboxes.append(box)
            self.timestamps.append(t)
        self.start,self.end = min(self.timestamps), max(self.timestamps)
        self.oid = oid
        self.interaction_dict = defaultdict(list)
 
    def get_box_from_t(self,t):
        t = t-self.start
        if t>=0: return self.bboxes[t]

    def interaction_analysis(self, car):
        c_start, c_end = car.start, car.end
        tiou = float(min(self.end,c_end)-max(self.start,c_start))/(max(c_end, self.end)-min(self.start, c_start))
        if tiou<=0: return
        start, end = max(self.start,c_start), min(self.end,c_end)
        for t in range(start,end+1):
            box1,box2 = self.get_box_from_t(t), car.get_box_from_t(t)
            iou_value = iou(box1,box2)
            if iou_value>0.001:
                self.interaction_dict[t].append(car.oid)

"""
    def motion_analysis(self):
    def interaction_analysis(self, car_st_boxes):
    def reid_analysis(self, person_st_boxes):
"""

"""
fit the requirement of ActEV output format
"""
def output_activity(actcount, name='Opening', score=0.5, vid='', start=0, end=1):
    act_out = {}
    act_out['activity'] = name
    act_out['activityID'] = actcount
    act_out['presenceConf'] = score
    act_out['alertFrame'] = int(end)
    act_out['localization'] = {}
    act_out['localization'][vid+'.mp4'] = {str(start):1, str(end):0}
    return act_out

def person_reid(person1, person2):
    s1,e1,s2,e2 = person1.start, person1.end, person2.start, person2.end
    tiou = float(min(e1,e2)-max(s1,s2))/(max(e1, e2)-min(s1, s2))
    if tiou>0: return -1
    if person1.start>person2.start:
        person1, person2 = person2, person1
    temp_dist = max(s1,s2)-min(e1,e2)
    if temp_dist>60: return -1
    box1, box2 = person1.bboxes[-1], person2.bboxes[0]
    s_dist = spatial_dist(box1,box2)
    iou_value = iou(box1, box2)
    reid_score = np.linalg.norm(person1.appearance-person2.appearance)
    return 1-reid_score/10

def spatial_dist(box1, box2):
    xa1,ya1,xa2,ya2 = box1
    xb1,yb1,xb2,yb2 = box2
    xa = (xa1+xa2)/2
    ya = (ya1+ya2)/2
    xb = (xb1+xb2)/2
    yb = (yb1+yb2)/2
    return np.sqrt((xa-xb)**2+(ya-yb)**2)
    
