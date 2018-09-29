import cv2
import numpy as np
from vehicle import Vehicle
from utils.utils import *
import pickle


def main():
    counter = 0  # to break after counter number of frames
    window_size = 2
    vehicles = {}
    with open("../data/car.txt", "r") as f:
        for line in f:
            counter += 1
            if counter > 847:
                break
            val = line.strip().split(",")
            frame_num = int(val[0])
            # store the vehicle ids, frame numbers with points
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
                    slope = get_vector((x, y), v, frame_num, window_size)
                    v.init_slope = slope
                    continue

            slope = get_vector((x, y), v, frame_num, window_size)
            theta = angle_between(slope, v.init_slope)
            v.theta.append(theta)
            v.slopes.append(slope)

    for v_id in vehicles:
        print(v_id)
        print(max(vehicles[v_id].theta))
        hist_bin(vehicles[v_id].theta)
        pickle.dump(vehicles[v_id].slopes, open('../data/slopes.pkl', 'wb'))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
