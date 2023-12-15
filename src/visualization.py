import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_data(data, title='Data', fs=10000, action='save'):
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    num_channels = data.shape[1]
    time_vector = np.arange(len(data)) / fs
    
    # Set the figure size here
    plt.figure(figsize=(12, num_channels*2))

    for i in range(num_channels):
        plt.subplot(num_channels, 2, i + 1)
        plt.plot(time_vector, data[:, i])  # Ensure data is plotted correctly per channel
        plt.ylabel(f'Ch {i+1}')
        plt.xlabel('Time (s)')

    plt.suptitle(f'{title}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if action == 'show':
        # Show the plot
        #plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()
    else:
        # Save the plot
        #plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Ensure the filename is valid
        plt.close()

