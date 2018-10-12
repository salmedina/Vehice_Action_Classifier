import argparse
import os
from operator import attrgetter

import seaborn as sns
from Car import Car
from matplotlib import pyplot as plt
from utils.utils import load_tracking_data

sns.set_style("ticks")

def render_plots_for_video(cars, car_id, window_size, norm_save_path, angle_save_path):
    car = cars[car_id]
    state_list = car.get_state_list(window_size)
    state_list = sorted(state_list, key=attrgetter('frame'))
    norm_list = [s.norm for s in state_list]
    angle_list = [s.angle for s in state_list]
    for i in range(len(state_list)):
        state = state_list[i]
        # Norm
        plt.title('Norm id@%d  ws@%d'%(car_id, window_size))
        plt.plot(norm_list)
        plt.axvline(x=i, color='#FFA500')
        plt.axhline(y=30, color='red')
        plt.xlabel('Frame')
        plt.ylabel('Distance [Pixels]')
        plt.savefig(os.path.join(norm_save_path, '%06d.png'%(state.frame)))
        plt.clf()
        # Angle
        plt.title('Angle id@%d  ws@%d'%(car_id, window_size))
        plt.plot(angle_list)
        plt.axvline(x=i, color='#FFA500')
        plt.xlabel('Frame')
        plt.ylabel('Degrees')
        plt.savefig(os.path.join(angle_save_path, '%06d.png' % (state.frame)))
        plt.clf()

        if i%100 == 0:
            print('Rendered up to frame %d'%(i))


def plot_cars_norm(cars, window_sizes):
    for cid in cars.keys():
        car = cars[cid]
        plt.title('Norm id@%d' % (cid))
        for ws in window_sizes:
            plt.plot(car.get_attr_list('norm', ws), label='%d frames'%(ws))
        plt.xlabel('Frame Index')
        plt.ylabel('Distance in Pixels')
        plt.legend()
        plt.savefig('../output/NormPlot_id_%d.png' % (cid))
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
        plt.savefig('../output/AnglePlot_id_%d.png'%(cid))
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
        plt.savefig('../output/MovingPlot_id_%d.png' % (cid))
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
    render_plots_for_video(cars, 1, 60, '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Norm/', '/Volumes/Seagate Backup Plus Drive/OUTPUT/VIRAT/VIRAT_S_000000/Angles/')
    # plot_cars_moving(cars, window_sizes)
    # plot_cars_norm(cars, window_sizes)
    # plot_cars_angles(cars, window_sizes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifies if tracklets are moving')
    parser.add_argument('-td', dest='trackingData', help='Path to the tracking data file')
    parser.add_argument('-ws', dest='windowSizes', type=str, default=60, help='Window size to detect vehicle movement')
    args = parser.parse_args()

    # main(args.trackingData, args.windowSize)
    window_sizes = [int(x) for x in args.windowSizes.strip().split(',')]
    main_analysis(args.trackingData, window_sizes)