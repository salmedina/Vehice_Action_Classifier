import argparse
import os.path as osp
from PIL import Image, ImageDraw
import mrcnn.model as mrcnnlib
from coco import coco
from glob import glob
import numpy as np
from scipy.spatial import distance
import pdb


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

MODEL_DIR = '../models'
COCO_MODEL_FILENAME = 'mask_rcnn_coco.h5'

def get_centroid(bbox):
    x0, y0, x1, y1 = bbox
    return (x1-x0)//2, (y1-y0)//2


def get_closest_bbox_to_image_center(image, bboxes):
    Ci = (image.width//2, image.height//2)
    bbox_centroids = [get_centroid(bb) for bb in bboxes]
    return bboxes[np.argmin([distance.euclidean(c, Ci) for c in bbox_centroids])]


def get_biggest_bbox(bboxes):
    return np.argmax([(x1-x0)*(y1-y0) for x0, y0, x1, y1 in bboxes])


def init_maskrcnn(model_dir, model_filename):
    config = InferenceConfig()
    model = mrcnnlib.MaskRCNN(mode='inference', model_dir=model_dir, config=config)
    model.load_weights(osp.join(model_dir, model_filename), by_name=True)
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    return model, class_names


def get_anno_line(image_path, bbox):
    filename = osp.basename(image_path)
    x0, y0, x1, y1 = bbox
    return f'{filename},{x0},{y0},{x1},{y1}'


def draw_bbox(image, bbox):
    canvas = ImageDraw.Draw(image)
    canvas.rectangle(bbox, fill=None, outline=(255, 255, 0), width=3)
    del canvas
    return image


def main(images_dir, output_dir, anno_filename, image_ext):
    model, labels = init_maskrcnn(MODEL_DIR, COCO_MODEL_FILENAME)
    vehicle_labels = [3, 6, 8]  #car, bus, truck

    image_path_list = glob(osp.join(images_dir, f'*{image_ext}'))

    output_lines = []

    for image_path in image_path_list:
        print(f'Image: {image_path}')
        image = Image.open(image_path)
        image_array = np.array(image)
        detect_res = model.detect([image_array])[0]
        vehicle_bboxes = [[x0, y0, x1, y1] for (y0, x0, y1, x1), label in zip(detect_res['rois'], detect_res['class_ids']) if label in vehicle_labels]
        if len(vehicle_bboxes) > 1:
            print(f'Detected vehicles: {len(vehicle_bboxes)}')
            biggest_bbox = get_closest_bbox_to_image_center(image, vehicle_bboxes)
            output_lines.append(get_anno_line(image_path, biggest_bbox))
            bboxed_image = draw_bbox(image, biggest_bbox)
            bboxed_image.save(osp.join(output_dir, osp.basename(image_path)))

    with open(osp.join(output_dir, anno_filename), 'w') as fout:
        fout.write('\n'.join(output_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory with the input images')
    parser.add_argument('--output_dir', type=str, help='Output directory where vehicle images will be stored')
    parser.add_argument('--anno_filename', type=str, default='vehicle_bbox.csv', help='Output directory where vehicle images will be stored')
    parser.add_argument('--img_ext', type=str, default='.jpg', help='Image file extension to be processed')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.anno_filename, args.img_ext)
