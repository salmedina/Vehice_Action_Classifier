import pycocotools.mask as maskUtils
import json
import numpy as np
from easydict import EasyDict as edict

def calc_iou(box1, box2):
    # Transform to two point bbox [x1, y1, x2, y2] instead of [x, y, w, h]
    box1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
    box2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def extract_car_segmentations(json_path):
    '''
    Data in files is a list of dicts with the following keys:
    'segmentation': {'counts','size'}
    'category_id': int
    'score': float
    'cat_name': string
    'bbox' : [x, y, w, h]
    :param json_path:
    :return:
    '''

    data = json.load(open(json_path))
    car_detections = []
    for detection in data:
        if detection['cat_name'] == 'car':
            car_seg = edict(dict(score=0.0, bbox=[], mask=None, size=(0, 0)))
            car_seg.score = detection['score']
            car_seg.bbox = [int(x) for x in detection['bbox']]
            car_seg.mask = maskUtils.decode(detection['segmentation'])
            car_seg.size = detection['segmentation']['size']
            car_detections.append(car_seg)

    return car_detections

def find_segmentation(query_bbox, json_path):
    car_segmentations = extract_car_segmentations(json_path)
    match_idx = np.argmax([calc_iou(query_bbox, seg['bbox']) for seg in car_segmentations])
    return car_segmentations[match_idx]