import cv2
from vehicle import Vehicle, Global_Variables
from utils.utils import *
# import pickle


def main():
    counter = 0  # to break after counter number of frames
    window_size = 15
    vehicles = {}
    global_variables = Global_Variables()
    with open("../data/car.txt", "r") as f:
        for line in f:
            counter += 1
            if counter < 0:
                continue
            # if counter > 500:
                # break
            val = line.strip().split(",")
            frame_num = int(val[0])
            # store the vehicle ids, frame numbers with points
            vehicle_id = val[1]
            x = int(val[2]) + int(val[4])/2
            y = int(val[3]) + int(val[5])/2

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
                    vector, norm = get_vector((x, y), v, frame_num, window_size)
                    v.init_slope = vector
                    global_variables.max_norm = max(norm, global_variables.max_norm)
                    print(vector)
                    continue

            vector, norm = get_vector((x, y), v, frame_num, window_size)
            theta = angle_between(vector, v.init_slope)
            v.theta.append(theta)
            v.vectors.append(vector)
            v.norms.append(norm)
            global_variables.max_norm = max(norm, global_variables.max_norm)

    for v_id in vehicles:
        try:
            print(v_id)
            print(max(vehicles[v_id].theta))
            print(global_variables.max_norm)
            get_plot([i/global_variables.max_norm for i in vehicles[v_id].norms])
        except:
            continue
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
