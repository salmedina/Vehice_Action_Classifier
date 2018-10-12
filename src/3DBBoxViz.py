import argparse
import os
from collections import namedtuple

import cv2
import numpy as np
from utils.box import build_box_3d, draw_box_3d
from utils.segmentation import find_segmentation
from utils.utils import load_tracking_data

Calibration = namedtuple('Calibration', ['image_vp1', 'image_vp2', 'image_vp3'])
Color = namedtuple('Color', ['r', 'g', 'b'])

def get_segmentation_filename(segmentation_dir, video_name, frame_num):
    return os.path.join(segmentation_dir, video_name, '_F_%8d.json'%frame_num)



def colorize_mask(mask, size, color):
    'mask comes in 255 based value and color as (r,g,b) tuple'
    color_mask = np.zeros([size[0], size[1], 3])
    color_mask[:, :, 0] = mask * color.b
    color_mask[:, :, 1] = mask * color.g
    color_mask[:, :, 2] = mask * color.r

    return color_mask

def main_proto():
    car_id = 1
    tracking_txt = '/home/zal/Devel/Vehice_Action_Classifier/data/car.txt'
    vps = [(6509, -770), (960, 1430), (-682, -770)]
    calib = Calibration(vps[0], vps[1], vps[2])
    draw_cache_dir = '/home/zal/Data/VIRAT/Output/3DBBox/VIRAT_S_000000/%03d' % car_id
    tracking_data = load_tracking_data(tracking_txt)
    output_video_path = '../output/3dBBox.mp4'
    drawn_files = []

    print('Drawing 3D Bounding Boxes')
    for tracklet in tracking_data[car_id]:
        tracklet_bbox = [tracklet.x, tracklet.y, tracklet.w, tracklet.h]

        frame_file = '/home/zal/Data/VIRAT/Frames/VIRAT_S_000000/%06d.png' % tracklet.frame
        seg_json = '/home/zal/Data/VIRAT/Output/objdetector/train/VIRAT_S_000000_F_%08d.json' % tracklet.frame
        drawn_frame_path = os.path.join(draw_cache_dir, '%06d.png' % tracklet.frame)

        segmentation = find_segmentation(tracklet_bbox, seg_json)

        frame_img = cv2.imread(frame_file)
        color_mask = colorize_mask(segmentation.mask, (segmentation.size[0], segmentation.size[1]), Color(138,43,226))
        frame_img = cv2.addWeighted(frame_img, 1.0, color_mask.astype(np.uint8), 0.5, 0)

        contour_img = np.zeros([segmentation.size[0], segmentation.size[1], 1])
        contour_img[:, :, 0] = segmentation.mask * 255

        box_3d = build_box_3d(contour_img, calib)
        draw_box_3d(frame_img, box_3d)

        cv2.imwrite(drawn_frame_path, frame_img)
        drawn_files.append(drawn_frame_path)

        if len(drawn_files) % 100 == 0:
            print('Processed frames: %d / %d' % (len(drawn_files), len(tracking_data[car_id])))

    print('Rendering video')
    frame_height, frame_width = cv2.imread(drawn_files[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, 29.97, (frame_width, frame_height), True)

    for frame_path in drawn_files:
        output_video.write(cv2.imread(frame_path))
    output_video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Tracker BBox')
    parser.add_argument('-vf', dest='videoFrames', help='Directory where the video frames are located')
    parser.add_argument('-td', dest='trackingData', help='Path to the tracking data file')
    parser.add_argument('-m', dest='masks', help='Directory with the processed Mask-RCNN output')
    parser.add_argument('-o', dest='output', help='Path to the output video file')
    parser.add_argument('-tid', dest='trackletId', type=int, help='Tracklet ID which wants to be visualized')
    args = parser.parse_args()

    #main(args.videoFrames, args.trackingData, args.masks, args.output, args.trackletId)
    main_proto()