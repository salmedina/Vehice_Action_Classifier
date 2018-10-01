import argparse
from Car import Car
from matplotlib import pyplot as plt
from utils.utils import load_tracking_data

def plot_cars_norm(cars, window_sizes):
    for cid in cars.keys():
        car = cars[cid]
        plt.title('Norm id@%d' % (cid))
        for ws in window_sizes:
            plt.plot(car.get_attr_list('norm', ws), label='%d frames'%(ws))
        plt.xlabel('Frame Index')
        plt.ylabel('Distance in Pixels')
        plt.legend()
        plt.show()
        plt.clf()

def plot_cars_angles(cars, window_sizes):
    for cid in cars.keys():
        car = cars[cid]
        plt.title('Angles id@%d' % (cid))
        for ws in window_sizes:
            plt.plot(car.get_attr_list('angle', ws), label='%d frames'%(ws))
        plt.xlabel('Frame Index')
        plt.ylabel('Degrees')
        plt.legend()
        plt.show()
        plt.clf()

def plot_cars_moving(cars, window_sizes):
    for cid in cars.keys():
        car = cars[cid]
        plt.title('Moving id@%d' % (cid))
        for ws in window_sizes:
            plt.plot(car.get_attr_list('moving', ws), label='%d frames'%(ws))
        plt.xlabel('Frame Index')
        plt.ylabel('Moving Impulse')
        plt.legend()
        plt.show()
        plt.clf()

def calc_cars_data(tracking_data, window_sizes):
    cars = {}

    for tid in tracking_data.keys():
        for ws in window_sizes:
            if len(tracking_data[tid]) < ws:
                continue
            cars[tid] = Car()
            cars[tid].calc_states(tracking_data[tid], ws, 30)

    return cars

def main_analysis(tracking_data_path, window_sizes):
    tracking_data = load_tracking_data(tracking_data_path)
    cars = calc_cars_data(tracking_data, window_sizes)
    plot_cars_moving(cars, window_sizes)
    plot_cars_norm(cars, window_sizes)
    plot_cars_angles(cars, window_sizes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifies if tracklets are moving')
    parser.add_argument('-td', dest='trackingData', help='Path to the tracking data file')
    parser.add_argument('-ws', dest='windowSizes', type=str, default=60, help='Window size to detect vehicle movement')
    args = parser.parse_args()

    # main(args.trackingData, args.windowSize)
    window_sizes = [int(x) for x in args.windowSizes.strip().split(',')]
    main_analysis(args.trackingData, window_sizes)