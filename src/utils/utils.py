import numpy as np
from matplotlib import pyplot as plt
import math


def hist_bin(data):
    bins = np.arange(-100, 100, 0.1)
    plt.xlim([min(data)-5, max(data)+5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('Slopes of the car')
    plt.xlabel('Degrees')
    plt.ylabel('Frequency')
    plt.show()


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
    return unit_vector


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    return math.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
