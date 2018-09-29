import numpy as np
from matplotlib import pyplot as plt


def hist_bin(data):
    bins = np.arange(-100, 100, 0.1)
    plt.xlim([min(data)-5, max(data)+5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('Slopes of the car')
    plt.xlabel('Degrees')
    plt.ylabel('Frequency')
    plt.show()