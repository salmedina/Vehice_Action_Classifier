from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from bs4 import BeautifulSoup
import cv2


def get_int(val):
    return int(float(val))


def parse_xml_file(filename):
    f = open(filename, 'r').read()
    e = BeautifulSoup(f, features="html5lib")
    # frames in a video
    frames = e.findAll('target_list')
    boxes = []
    orientations = []
    for frame in frames:
        vehicles = frame.findAll('target')
        frame_boxes = []
        frame_orientations = []
        for vehicle in vehicles:
            frame_boxes.append(vehicle.box.attrs)
            frame_orientations.append(vehicle.attribute.attrs)
        boxes.append(frame_boxes)
        orientations.append(frame_orientations)
    return boxes, orientations


dir_path = '/home/hima/hima/data/Insight-MVT_Annotation_Train/'
xml_path = './xml_data/'
resnet_box_features = '/home/hima/hima/data/resnet_box_features/'
resnet_model = ResNet50(weights='imagenet', include_top=False)

# get resnet features
for dir_ in os.listdir(dir_path):
    tmp_dir = dir_path + dir_ + "/"
    print(tmp_dir)
    xml_file = xml_path + dir_ + ".xml"
    boxes, orientations = parse_xml_file(xml_file)
    for file in os.listdir(tmp_dir):
        img = cv2.imread(tmp_dir + file)
        frame_num = int(file[-9: -4])
        frame_boxes = boxes[frame_num - 1]
        frame_orientations = orientations[frame_num - 1]
        for i, box in enumerate(frame_boxes):
            img_rows = img[get_int(box['top']):get_int(box['top'])+get_int(box['height'])]
            final_img = img_rows[:, get_int(box['left']):get_int(box['left'])+get_int(box['width'])]
            # pre_data = preprocess_input(np.expand_dims(final_img, axis=0))
            resized_img = cv2.resize(final_img, (244, 244))
            r_f = resnet_model.predict(resized_img.reshape(1, 244, 244, 3)) # input
            output = orientations[i]
            np.save(open(resnet_box_features+dir_+"|"+file+"|"+str(i), 'w'), r_f)
