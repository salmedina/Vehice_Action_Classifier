import pycocotools.mask as maskUtils
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def bb_intersection_over_union(box1, box2):
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

def extract_car_masks_from_json(json_path):
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
    car_masks = []
    for det in data:
        if det['cat_name'] == 'car':
            car_mask = {}
            car_mask['score'] = det['score']
            car_mask['bbox'] = det['bbox']
            car_mask['mask'] = maskUtils.decode(det)

    return car_masks

def get_box_mask(box, json_path):
    car_masks = extract_car_masks_from_json(json_path)

def main_proto():
    start_pos, end_pos = 1500, 2000

    for i in range(start_pos, end_pos):
        filename = '/Users/zal/CMU/Projects/DIVA/Data/ObjectDetector/VIRAT_S_000000_F_%08d.json' % i
        car_segmentations = extract_car_masks_from_json(filename)
        if len(car_segmentations) > 0:
            masks = [s['mask'] for s in car_segmentations]
            compound_mask = masks[0]
            for m in masks[1:]:
                compound_mask = np.logical_and(compound_mask, m)

        Image.fromarray(compound_mask).save('/Users/zal/CMU/Projects/DIVA/Data/ObjectDetector/imgs/VIRAT_S_000000_F_%08d.png' % i)
        break

if __name__ == '__main__':
    main_proto()