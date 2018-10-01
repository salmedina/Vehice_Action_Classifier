import math
import numpy as np
from collections import namedtuple
from operator import attrgetter
from matplotlib import pyplot as plt

Tracklet = namedtuple('Tracklet', ['frame', 'x', 'y', 'w', 'h', 'cx', 'cy'])

def hist_bin(data):
    bins = np.arange(-100, 100, 0.1)
    plt.xlim([min(data)-5, max(data)+5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('Slopes of the car')
    plt.xlabel('Degrees')
    plt.ylabel('Frequency')
    plt.show()

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


def plot_data(data, title="Angle of the vehicle", xlabel="Frame", ylabel="Norm", max_y=100):
    plt.title(title)
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim([0, 1])
    plt.show()
    plt.clf()


def calc_vector(p1, p2):
    '''Calculates the vector from p1 to p2'''
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    norm = math.sqrt(delta_x*delta_x + delta_y*delta_y)
    unit_vector = (0,0) if norm == 0 else (delta_x/norm, delta_y/norm)

    return unit_vector, norm

def get_vector(pt1, v, frame_num, window_size):
    (x, y) = pt1
    delta_y = y - v.car_points[frame_num-window_size][1]
    delta_x = x - v.car_points[frame_num-window_size][0]
    distance = [delta_x, delta_y]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
    if norm == 0:
        unit_vector = [0, 0]
    else:
        unit_vector = [distance[0] / norm, distance[1] / norm]
    return unit_vector, norm


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    val = math.degrees(np.arccos(np.dot(v1, v2)))
    if val == 180.0:
        print(v1, v2)
    return val
