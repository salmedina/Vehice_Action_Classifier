import argparse
import csv
from os.path import join, basename, splitext

import cv2
import numpy as np
from PIL import Image


def rotate_max_area(image, angle):
    '''
    Rotates an image and crops to the max visible area without black borders
    :param image: cv2 image
    :param angle: angle in degrees, can be positive or negative
    :return: rotated image with cropped borders
    '''
    def get_rotated_rect_max_area(w, h, angle):
        if w <= 0 or h <= 0:
            return 0,0

        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            # two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
            # crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr,hr

    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    wr, hr = get_rotated_rect_max_area(image.shape[1], image.shape[0], math.radians(angle))
    rotated = rotate_bound(image, angle)
    h, w, _ = rotated.shape
    y1 = h//2 - int(hr/2)
    y2 = y1 + int(hr)
    x1 = w//2 - int(wr/2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]

def resize_and_pad(img_path, target_size):
    img = Image.open(img_path)
    new_img = Image.new('RGB', (target_size, target_size), (119,118,120))
    img_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(target_size) / max(img_size)
    new_size = tuple([int(x * ratio) for x in img_size])
    new_img.paste(img.resize(new_size, Image.BILINEAR),
                  ((target_size - new_size[0]) // 2,
                    (target_size - new_size[1]) // 2))
    return new_img

def get_mirror_name(img_filename):
    name, ext = splitext(basename(img_filename))
    return '%s_lr%s'%(name, ext)

def mirror_sample(img, orientation, bin_idx, nbins=16):
    return img.transpose(Image.FLIP_LEFT_RIGHT), (360-orientation)%360, (nbins-1)-bin_idx

def main(data_dir, labels_path, output_size, output_dir):
    reader = csv.reader(open(labels_path))
    next(reader, None)

    print('Augmenting data')
    augmented_data = []
    counter = 0
    for csv_entry in reader:
        img_filename, orientation, bin_idx = csv_entry

        source_img_path = join(data_dir, img_filename)
        target_img_path = join(output_dir, img_filename)
        resized_image = resize_and_pad(source_img_path, output_size)
        resized_image.save(target_img_path)
        augmented_data.append('%s,%s,%s'%(img_filename, orientation, bin_idx))

        mirror_img, mirror_orientation, mirror_bin = mirror_sample(resized_image, int(orientation), int(bin_idx))
        mirror_img_filename = get_mirror_name(img_filename)
        mirror_img.save(join(output_dir, mirror_img_filename))
        augmented_data.append('%s,%d,%d'%(mirror_img_filename, mirror_orientation, mirror_bin))
        counter +=1
        if counter %1000 == 0:
            print('Processed entries:  %d'% counter)

    print('Saving annotations')
    output_file = open(join(output_dir, basename(labels_path)), 'w')
    output_file.writelines('\n'.join(augmented_data) + '\n')
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesses the extracted orientation data')
    parser.add_argument('-dd', dest='dataDir', help='Directory with the orientation data')
    parser.add_argument('-l', dest='labelsPath', help='Labels file path')
    parser.add_argument('-sz', dest='size', type=int, help='Target size, the output images will be squared')
    parser.add_argument('-td', dest='targetDir', help='Directory that stores preprocessed data')
    args = parser.parse_args()

    main(args.dataDir, args.labelsPath, args.size, args.targetDir)