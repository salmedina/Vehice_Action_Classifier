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
resnet_model = ResNet50(weights='imagenet', include_top=False)

for dir_ in os.listdir(dir_path):
    tmp_dir = dir_path + dir_ + "/"
    print(tmp_dir)
    xml_file = xml_path + dir_ + ".xml"
    boxes, orientations = parse_xml_file(xml_file)
    import pdb; pdb.set_trace()
    for file in os.listdir(tmp_dir):
        img = image.img_to_array(image.load_img(tmp_dir + file))
        frame_num = int(file[-9: -4])
        frame_boxes = boxes[frame_num - 1]
        frame_orientations = orientations[frame_num - 1]
        for i, box in enumerate(frame_boxes):
            img_rows = img[get_int(box['top']):get_int(box['top'])+get_int(box['height'])]
            final_img = img_rows[:, get_int(box['left']):get_int(box['left'])+get_int(box['width'])]
            print(final_img, frame_orientations[i])
            # cv2.imshow(final_img)
            # img_data = image.img_to_array(final_img)
            # img_data = np.expand_dims(img_data, axis=0)
            # img_data = preprocess_input(img_data)
            # resnet_feature = resnet_model.predict(img_data)
