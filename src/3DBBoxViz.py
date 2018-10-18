import argparse
import os
from collections import namedtuple

import cv2
import numpy as np
from utils.box import build_box_3d, draw_box_3d
from utils.segmentation import find_segmentation
from utils.utils import load_tracking_data
from utils.utils import Tracklet

from PIL import Image

Point = namedtuple('Point', ['x','y'])
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

def main_frame():
    car_id = 1
    tracking_txt = '/home/zal/Data/VIRAT/tracklets/VIRAT_S_000001/car.txt'
    # vps = [(6509, -770), (960, 1430), (-682, -770)]
    vps = [(2013, -300), (-3193, 97), (1034, 2952)]

    calib = Calibration(vps[0], vps[1], vps[2])
    draw_cache_dir = '/home/zal/Data/VIRAT/Output/3DBBox/VIRAT_S_000001/%03d' % car_id
    tracking_data = load_tracking_data(tracking_txt)

    print('Drawing 3D Bounding Boxes')

    tracklet = Tracklet(3, 622, 466, 249, 186, 746, 559)
    tracklet_bbox = [tracklet.x, tracklet.y, tracklet.w, tracklet.h]

    frame_file = '/home/zal/Data/VIRAT/Frames/VIRAT_S_000001/%06d.jpg' % tracklet.frame
    seg_json = '/home/zal/Data/VIRAT/Output/objdetector/train/VIRAT_S_000001_F_%08d.json' % tracklet.frame
    drawn_frame_path = os.path.join(draw_cache_dir, '%06d.jpg' % tracklet.frame)

    segmentation = find_segmentation(tracklet_bbox, seg_json)

    frame_img = cv2.imread(frame_file)
    color_mask = colorize_mask(segmentation.mask, (segmentation.size[0], segmentation.size[1]), Color(138,43,226))
    drawn_img = cv2.addWeighted(frame_img, 1.0, color_mask.astype(np.uint8), 0.5, 0)
    cv2.imwrite('/home/zal/Data/VIRAT/Output/3DBBox/VIRAT_S_000001/seg_%06d.jpg'% tracklet.frame, drawn_img)

    contour_img = np.zeros([segmentation.size[0], segmentation.size[1], 1])
    contour_img[:, :, 0] = segmentation.mask * 255

    box_3d = build_box_3d(contour_img, calib)
    box_3d.points.a = [775, 454]
    box_3d.points.c = [696,582]
    box_3d.points.g = [699, 668]
    box_3d.points.h = [612, 638]
    draw_box_3d(drawn_img, box_3d)

    cv2.imwrite(drawn_frame_path, drawn_img)
    cv2.imshow('', drawn_img)
    cv2.waitKey(0)

    # Unwarp car into image
    ss = 100
    ls = 270
    pts = box_3d.points

    front_pts = np.float32([pts.d, pts.c, pts.g, pts.h])
    front_warp_pts = np.float32([[0, 0], [ss, 0], [ss, ss], [0, ss]])
    M = cv2.getPerspectiveTransform(front_pts, front_warp_pts)
    front_img = cv2.warpPerspective(frame_img, M, (ss, ss))

    side_pts = np.float32([pts.c, pts.b, pts.f, pts.g])
    side_warp_pts = np.float32([[0, 0], [ls, 0], [ls, ss], [0, ss]])
    M = cv2.getPerspectiveTransform(side_pts, side_warp_pts)
    side_img = cv2.warpPerspective(frame_img, M, (ls, ss))

    top_pts = np.float32([pts.d, pts.a, pts.b, pts.c])
    top_warp_pts = np.float32([[0, 0], [ls, 0], [ls, ss], [0, ss]])
    M = cv2.getPerspectiveTransform(top_pts, top_warp_pts)
    top_img = cv2.warpPerspective(frame_img, M, (ls, ss))

    unwrapped_img = Image.new('RGB', (ss+ls, 2*ss))
    unwrapped_img.paste(Image.fromarray(front_img) , (0, ss))
    unwrapped_img.paste(Image.fromarray(top_img) , (ss, 0))
    unwrapped_img.paste(Image.fromarray(side_img) , (ss, ss))
    unwrapped_img.save('/home/zal/Data/VIRAT/Output/3DBBox/VIRAT_S_000001/unwrap_%06d.jpg'% tracklet.frame)


def main_proto():
    car_id = 1
    tracking_txt = '/home/zal/Data/VIRAT/tracklets/VIRAT_S_000001/car.txt'
    vps = [(6509, -770), (960, 1430), (-682, -770)]
    calib = Calibration(vps[0], vps[1], vps[2])
    draw_cache_dir = '/home/zal/Data/VIRAT/Output/3DBBox/VIRAT_S_000001/%03d' % car_id
    tracking_data = load_tracking_data(tracking_txt)
    output_video_path = '../output/3dBBox_000001.mp4'
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
    # main_proto()
    main_frame()