import cv2
import numpy as np
from vehicle import Vehicle
import math
from utils import *


start = 1
tmp = 0
images = []
counter = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter()
out = vout.open('output.mp4', fourcc, 29, (1898, 688), True)
img = 0
c = 0
window_size = 2
patience_reset_init_slope = 4
vehicles = {}
delta = 0.01


def get_slope(v, frame_num, window_size):
    delta_y = y-v.car_points[frame_num-window_size][1]
    delta_x = x-v.car_points[frame_num-window_size][0]
    if delta_x == 0:
        slope = -1e-100
    else:
        slope = delta_y/delta_x
    return slope


def angle_bw_lines(m1, m2):
    theta = math.degrees(math.atan(abs(m1 - m2)/(1+m1*m2)))
    # print("angle ", theta, math.atan(abs(m1 - m2)/(1+m1*m2)), m1, m2)
    return theta


with open("./car.txt", "r") as f:
    for line in f:
        counter += 1
        if counter > 847:
            break
        val = line.strip().split(",")
        frame_num = int(val[0])

        if frame_num != tmp and not start:
            cv2.imwrite('color_img.png', img)
            frame = cv2.imread('color_img.png')
            vout.write(frame)
            img = np.zeros((688, 1898, 3), np.uint8)
        elif start:
            img = np.zeros((688, 1898, 3), np.uint8)
            start = 0
        # store the vehicle ids, frame numbers with points
        c += 1
        vehicle_id = val[1]
        x = int(val[2]) + int(val[4])//2
        y = int(val[3]) + int(val[5])//2

        # dont check for the first window points
        if vehicle_id not in vehicles:
            vehicles[vehicle_id] = Vehicle()
            vehicles[vehicle_id].num_of_frames_covered = 1
            vehicles[vehicle_id].car_points[frame_num] = (x, y)
            continue
        else:
            # get the initial slope until you reach the window size
            v = vehicles[vehicle_id]
            v.car_points[frame_num] = (x, y)
            v.num_of_frames_covered += 1
            if v.num_of_frames_covered <= window_size:
                continue
            elif v.num_of_frames_covered == window_size+1:
                slope = get_slope(v, frame_num, window_size)
                v.init_slope = slope
                continue

        slope = get_slope(v, frame_num, window_size)
        theta = angle_bw_lines(slope, v.init_slope)
        if theta > 45 or theta < -45:
            print(theta, frame_num)
        v.theta.append(theta)
        v.slopes.append(slope)
        # change the slope
        print(slope, frame_num, v.init_slope, theta)
        if len(v.theta) > patience_reset_init_slope:
            if abs(v.theta[-1] - v.init_slope) < delta:
                # v.init_slope = slope
                print(v.init_slope)
                print("resetting init slope!")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        tmp = frame_num

for v_id in vehicles:
    print(v_id)
    print(max(vehicles[v_id].theta))
    hist_bin(vehicles[v_id].theta)

vout.release()
cv2.destroyAllWindows()
