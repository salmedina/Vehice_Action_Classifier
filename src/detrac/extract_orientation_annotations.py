import argparse
import glob
import untangle
from os.path import join, basename, splitext
import multiprocessing as mp
from joblib import Parallel, delayed

def get_baseame_no_ext(full_path):
    return splitext(basename(full_path))[0]

def extract_orientation_data(xml_filepath):
    '''
    Extracts the orientation data for each bounding box in each frame
    :param xml_filepath: Path to the DETRAC annotation file
    :return: a list of annotations that have frame, x, y, w, h, orientation
    '''
    xml = untangle.parse(xml_filepath)

    data = []
    data.append('frame,id,x,y,w,h,orientation')
    for frame in xml.sequence.frame:
        frame_num = frame['num']
        for t in frame.target_list.target:
            doi = [frame_num, t['id'],t.box['left'], t.box['top'], t.box['width'], t.box['height'], t.attribute['orientation']]
            data.append(','.join(doi))

    return data

def process_xml(xml_filepath, output_dir):
    orientation_data = extract_orientation_data(xml_filepath)
    output_file = open(join(output_dir, '%s.csv' % (get_baseame_no_ext(xml_filepath))), 'w')
    output_file.writelines('\n'.join(orientation_data) + '\n')
    output_file.close()

def main(source_dir, output_dir):
    xml_file_list = glob.glob(join(source_dir, '*.xml'))
    Parallel(n_jobs=mp.cpu_count())(delayed(process_xml)(xml_filepath=f, output_dir=output_dir) for f in xml_file_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts orientation from DETRAC dataset')
    parser.add_argument('-s', dest='sourceDir', help='Source dir with DETRAC annotation files')
    parser.add_argument('-o', dest='outputDir', help='Output dir where the csv files will be stored')
    args = parser.parse_args()
    main(args.sourceDir, args.outputDir)