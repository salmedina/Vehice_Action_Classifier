import argparse
import glob
import csv
from PIL import Image
from os.path import join, basename, splitext
import multiprocessing as mp
from joblib import Parallel, delayed

def get_baseame_no_ext(full_path):
    return splitext(basename(full_path))[0]

def process_row(row, frames_dir, video_name, bin_degs, output_dir):
    frame_num, id, x, y, w, h, orientation = [round(float(entry)) for entry in row]
    frame_img_path = join(frames_dir, video_name, 'img%05d.jpg' % frame_num)
    frame_img = Image.open(frame_img_path)
    vehicle_img = frame_img.crop((x, y, x + w, y + h))
    vehicle_img_filename = '%s_F_%d_ID_%d.jpg' % (video_name, frame_num, id)
    vehicle_img_path = join(output_dir, vehicle_img_filename)
    vehicle_img.save(vehicle_img_path)
    orientation_bin = int(orientation % 360 / bin_degs)
    return '%s,%d,%d' % (vehicle_img_filename, orientation, orientation_bin)

def main(frames_dir, annotations_dir, output_dir, num_bins=16):
    bin_degs = 360 / num_bins
    labeled_data = []
    labeled_data.append('image_file,orientation_bin')
    for anno_filename in glob.glob(join(annotations_dir, '*.csv')):
        print('Processing ', anno_filename)
        video_name = get_baseame_no_ext(anno_filename)
        csv_reader = csv.reader(open(anno_filename, 'r'))
        next(csv_reader, None)

        video_data = Parallel(n_jobs=mp.cpu_count())(delayed(process_row)(row=row, frames_dir=frames_dir, video_name=video_name, bin_degs=bin_degs, output_dir=output_dir) for row in csv_reader)

        labeled_data = labeled_data + video_data

    print('Saving labeled data')
    output_file = open(join(output_dir, 'orientation_labels_b_%d.csv' % num_bins), 'w')
    output_file.writelines('\n'.join(labeled_data) + '\n')
    output_file.close()

    print('Fin')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepares orientation data to be used in a model')
    parser.add_argument('-fd', dest='framesDir', help='Directory with the video frames')
    parser.add_argument('-ad', dest='annoDir', help='Directory with the orientation annotations')
    parser.add_argument('-nb', dest='numBins', default=16, type=int, help='Number of orientation bins for labels')
    parser.add_argument('-o', dest='outputDir', help='Directory where extracted frames and labels will be stored')
    args = parser.parse_args()

    main(args.framesDir, args.annoDir, args.outputDir, args.numBins)