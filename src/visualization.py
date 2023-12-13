import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_data(data, title='Data',fs=10000):
    num_channels = data.shape[1]
    time_vector = np.arange(len(data)) / fs
    for i in range(num_channels):
        plt.subplot(num_channels, 2, i + 1)
        plt.plot(time_vector, analog_signal[:, i].magnitude.flatten())
        plt.ylabel(f'Ch {i+1}')
        plt.xlabel('Time (s)')

    plt.suptitle(f'{title}', fontsize=36)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

