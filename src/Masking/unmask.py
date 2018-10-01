from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import json

filename = "./mrcnn_101/VIRAT_S_050000_12_001591_001619_F_00000833.json"
data = json.load(open(filename))
n = len(data)
for i in range(n):
    val = maskUtils.decode(data[i])
