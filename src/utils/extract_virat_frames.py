import argparse
import cv2
from glob import glob
import os.path as osp
import os

def extract_frames(video_dir, frames_dir):
    video_path = osp.join(video_dir, 'video.mp4')
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        count += 1
        image_path = osp.join(frames_dir, "%06d.jpg"%count)
        cv2.imwrite(image_path, image)
        success, image = vidcap.read()

def main(source_dir, target_dir):
    dir_list = [item for item in glob(osp.join(source_dir, '*/')) if osp.isdir(item)]
    total_videos = len(dir_list)
    for idx, video_dir in enumerate(dir_list):
        print('[{}/{}] {}'.format(idx, total_videos, video_dir))
        video_frames_dir = osp.join(target_dir, osp.basename(osp.normpath(video_dir)))
        os.makedirs(video_frames_dir, exist_ok=True)
        extract_frames(video_dir, video_frames_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extracts the frames from all the VIRAT videos')
    parser.add_argument('-i', dest='sourceDir', help='Directory where the videos are located')
    parser.add_argument('-o', dest='targetDir', help='Directory where the frames will be stored')
    args = parser.parse_args()

    main(args.sourceDir, args.targetDir)