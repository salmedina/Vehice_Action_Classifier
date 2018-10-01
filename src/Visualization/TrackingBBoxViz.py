import argparse
import cv2
import glob
import os
import sys
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing as mp
from operator import attrgetter
from collections import namedtuple

Tracklet = namedtuple('Tracklet', ['frame', 'x', 'y', 'w', 'h', 'cx', 'cy'])

def clear_cache_dir(drawing_cache_dir):
    '''Deletes the image files from the given dir'''
    for file in glob.glob(os.path.join(drawing_cache_dir, '*.png')):
        os.remove(file)

def render_video_from_frames(frames_dir, drawing_cache_dir, save_path, start_frame=0, end_frame=np.inf, fps=29.97):
    '''Renders the video from an image path list'''
    start_filename = '%06d.png' % start_frame
    end_frame = int(os.path.splitext(max(os.listdir(frames_dir)))[0]) if end_frame == np.inf else end_frame
    end_filename = '%06d.png' % end_frame

    # Get the list of all images in frames dir
    frame_filename_list = [filename for filename in os.listdir(frames_dir) if filename > start_filename and filename < end_filename]

    # Get the list of all images in the cache
    drawn_filename_list = [filename for filename in os.listdir(drawing_cache_dir) if filename > start_filename and filename < end_filename]

    # Get the list of files
    output_frame_path_list = []
    for frame_pos in range(start_frame, end_frame+1):
        frame_pos_filename = '%06d.png'%frame_pos
        if frame_pos_filename in drawn_filename_list:
            output_frame_path_list.append(os.path.join(drawing_cache_dir, frame_pos_filename))
        elif frame_pos_filename in frame_filename_list:
            output_frame_path_list.append(os.path.join(frames_dir, frame_pos_filename))

    frame_height, frame_width = cv2.imread(os.path.join(frames_dir, frame_filename_list[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height), True)

    for frame_path in output_frame_path_list:
        output_video.write(cv2.imread(frame_path))
    output_video.release()

def draw_tracklet(tracklet, drawn_frames_set, draw_cache_dir, video_frames_dir, color):
    '''Draws the tracklet bounding box and centroid on its frame'''
    imread_dir = draw_cache_dir if tracklet.frame in drawn_frames_set else video_frames_dir
    frame_img = cv2.imread(os.path.join(imread_dir, '%06d.png' % tracklet.frame))
    cv2.rectangle(frame_img, (tracklet.x, tracklet.y), (tracklet.x + tracklet.w, tracklet.y + tracklet.h), color, 3)
    cv2.circle(frame_img, (tracklet.cx, tracklet.cy), 3, color, 2)
    cv2.imwrite(os.path.join(draw_cache_dir, '%06d.png' % tracklet.frame), frame_img)

def draw_tracking(video_frames_dir, tracking_data, draw_cache_dir, color_palette, id=None):
    '''Draws only the tracklets of the given id, if it is not given it draws all the tracklet in the tracking_data'''
    start_frame = np.inf
    end_frame = 0

    for id in tracking_data.keys() if id is None else [id]:
        print('Drawing tracklets for vehicle # %d'%id)
        drawn_frames_set = set(os.listdir(draw_cache_dir))
        Parallel(n_jobs=mp.cpu_count())(delayed(draw_tracklet)(tracklet = tracklet, drawn_frames_set=drawn_frames_set, draw_cache_dir=draw_cache_dir, video_frames_dir=video_frames_dir, color=color_palette[id]) for tracklet in tracking_data[id])
        min_frame = min(tracking_data[id], key=attrgetter('frame')).frame
        max_frame = max(tracking_data[id], key=attrgetter('frame')).frame
        start_frame = min_frame if min_frame < start_frame else start_frame
        end_frame = max_frame if max_frame > end_frame else end_frame

    return (start_frame, end_frame)

def build_color_palette(num_colors):
    '''Builds color palette from seaborn'''
    return [[int(ch*255) for ch in color] for color in sns.color_palette('Paired', num_colors)]

def get_frame_img_path_list(frames_dir):
    '''Returns sorted list of image paths'''
    frame_img_path_list = glob.glob(os.path.join(frames_dir, '*.png'))
    return frame_img_path_list.sort()

def load_tracking_data(tracking_data_path):
    '''Loads tracking data into dictionary of tracklets with vehicle id as key'''
    tracking_data = {}
    with open(tracking_data_path, 'r') as data_file:
        for line in data_file:
            frame, id, x, y, w, h, _, _, _, _ = [int(float(field)) for field in line.strip().split(',')]
            if id not in tracking_data:
                tracking_data[id] = []
            tracking_data[id].append(Tracklet(frame, x, y, w, h, x + int(w / 2), y + int(h / 2)))

    for id in tracking_data.keys():
        tracking_data[id] = sorted(tracking_data[id], key=attrgetter('frame'))

    return tracking_data

def main(video_frames_dir, tracking_data_path, drawing_cache_dir, output_video_path, tracklet_id):
    print('Loading tracking data')
    tracking_data = load_tracking_data(tracking_data_path)
    print('Building color palette')
    color_palette = build_color_palette(len(tracking_data.keys()))
    print('Clearing cache dir')
    clear_cache_dir(drawing_cache_dir)
    print('Drawing tracking')
    start_frame, end_frame = draw_tracking(video_frames_dir, tracking_data, drawing_cache_dir, color_palette, tracklet_id)
    print('Rendering output video')
    render_video_from_frames(video_frames_dir, drawing_cache_dir, output_video_path, start_frame, end_frame)

def validate_args(args):
    if not os.path.exists(args.videoFrames):
        print('Video frames dir does not exist:')
        print(args.videoFrames)
        return False
    if not os.path.exists(args.trackingData):
        print('Tracking data file does not exist')
        print(args.trackingData)
        return False
    if not os.path.exists(args.drawCacheDir):
        print('Drawing cache dir does not exist')
        print(args.drawCacheDir)
        return False
    return True

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualize Tracker BBox')
    parser.add_argument('-vf', dest='videoFrames', help='Directory where the video frames are located')
    parser.add_argument('-td', dest='trackingData', help='Path to the tracking data file')
    parser.add_argument('-dc', dest='drawCacheDir', help='Cache directory for drawn video frames ')
    parser.add_argument('-o', dest='output', help='Path to the output video file')
    parser.add_argument('-tid', dest='trackletId', required=False, type=int, default=None, help='Tracklet ID which wants to be visualized')
    args = parser.parse_args()

    if not validate_args(args):
        sys.exit(-1)

    main(args.videoFrames, args.trackingData, args.drawCacheDir, args.output, args.trackletId)