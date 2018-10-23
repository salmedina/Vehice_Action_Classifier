import argparse
import csv
import math
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

    image = pil_to_cv2(image)
    wr, hr = get_rotated_rect_max_area(image.shape[1], image.shape[0], math.radians(angle))
    rotated = rotate_bound(image, angle)
    h, w, _ = rotated.shape
    y1 = h//2 - int(hr/2)
    y2 = y1 + int(hr)
    x1 = w//2 - int(wr/2)
    x2 = x1 + int(wr)

    return Image.fromarray(cv2.cvtColor(rotated[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

def resize_and_pad(img, target_size):
    new_img = Image.new('RGB', (target_size, target_size), (119,118,120))
    img_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(target_size) / max(img_size)
    new_size = tuple([int(x * ratio) for x in img_size])
    new_img.paste(img.resize(new_size, Image.BILINEAR),
                  ((target_size - new_size[0]) // 2,
                    (target_size - new_size[1]) // 2))
    return new_img

def get_rotated_name(img_filename):
    name, ext = splitext(basename(img_filename))
    return '%s_rot%s'%(name, ext)

def random_rotate_image(img, orientation, nbins):
    angle_bound = 180/nbins
    rot_angle = np.random.uniform(-angle_bound, angle_bound, 1)[0]
    new_orientation = round((orientation + rot_angle)%360)
    return rotate_max_area(img, -rot_angle), new_orientation, int((new_orientation * nbins) / 360)

def get_mirror_name(img_filename):
    name, ext = splitext(basename(img_filename))
    return '%s_lr%s'%(name, ext)

def mirror_image(img, orientation, bin_idx, nbins):
    return img.transpose(Image.FLIP_LEFT_RIGHT), (360-orientation)%360, (nbins-1)-bin_idx

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main(data_dir, labels_path, output_size, output_dir, nbins):
    reader = csv.reader(open(labels_path))
    next(reader, None)

    print('Augmenting data')
    augmented_data = []
    counter = 0
    for csv_entry in reader:
        img_filename, orientation, bin_idx = csv_entry
        orientation = int(orientation)
        bin_idx = int(bin_idx)

        source_img_path = join(data_dir, img_filename)
        target_img_path = join(output_dir, img_filename)

        source_img = Image.open(source_img_path)
        resized_image = resize_and_pad(source_img, output_size)
        resized_image.save(target_img_path)
        augmented_data.append('%s,%s,%s'%(img_filename, orientation, bin_idx))

        # Calc and save the mirror image
        mirror_img, mirror_orientation, mirror_bin = mirror_image(source_img, orientation, bin_idx, nbins)
        resized_mirror_image = resize_and_pad(mirror_img, output_size)
        mirror_img_filename = get_mirror_name(img_filename)
        resized_mirror_image.save(join(output_dir, mirror_img_filename))
        augmented_data.append('%s,%d,%d'%(mirror_img_filename, mirror_orientation, mirror_bin))

        # Calc and rotate mirror/source image with 50% chance
        rot_source_img, rot_source_orientation = (source_img, orientation) if np.random.uniform(0.0, 1.0, 1)[0] < 0.5 else (mirror_img, mirror_orientation)

        rot_img, rot_orientation, rot_bin = random_rotate_image(rot_source_img, rot_source_orientation, nbins)
        rot_img = resize_and_pad(rot_img, output_size)
        rot_img_filename = get_rotated_name(img_filename)
        rot_img.save(join(output_dir, rot_img_filename))
        augmented_data.append('%s,%d,%d'%(rot_img_filename, rot_orientation, rot_bin))

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
    parser.add_argument('-sz', dest='size', type=int, help='Target size, the output images will be square images')
    parser.add_argument('-td', dest='targetDir', help='Directory that stores augmented data')
    parser.add_argument('-nb', dest='nbins', type=int, default=16, help='Number of orientation bins')
    args = parser.parse_args()

    main(args.dataDir, args.labelsPath, args.size, args.targetDir, args.nbins)