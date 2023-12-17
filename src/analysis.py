import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal

def plot_stft(data, nperseg, noverlap, window='blackman', fs=10000):
    f, t, Zxx = signal.stft(data, fs, nperseg, noverlap, window=f'{window}')
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    #plt.ylim(0,40)
    plt.xlabel('Time [sec]')
    plt.colorbar(f)
    plt.show()


def power_spectral_density(data, fs=10000, action='save'):
    for i in range(len(data[1])):
        (f, S) = signal.periodogram(data[i], fs, scaling='density', nfft=100)
        plt.semilogy(f, S)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        #plt.xlim(2,2000)
        #plt.ylim(10*1,10**9)
        plt.title(f"Channel {i+1} PSD")
        plt.show()

def psd_all(data, fs=10000, title='Data PSD', action='save'):
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    num_channels = data.shape[1]
    plt.figure(figsize=(10, 6))  # Adjust as needed

    for i in range(num_channels):
        (f, S) = signal.periodogram(data[:, i], fs, scaling='density', nfft=100)
        plt.semilogy(f, S, label=f'Ch {i+1}')

    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.ylim(0.0000001,1000)
    plt.title(title)
    plt.legend()

    if action == 'show':
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Ensure the filename is valid
        plt.close()



def plot_spectrogram_segments(data, fs, segment_length, nperseg, noverlap, action='save'):
    num_channels = data.shape[1]
    data_length = len(data)
    
    # Calculate the number of segments
    num_segments = int(np.ceil(data_length / segment_length))

    for chan_index in range(num_channels):
        for i in range(num_segments):
            start = i * segment_length
            end = min(start + segment_length, data_length)
            segment_data = data[start:end, chan_index]
            time_vector = np.arange(start, end) / fs

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(24, 8), sharex=True)

            # Raw Data Plot for the segment
            axs[0].plot(time_vector, segment_data)
            axs[0].set_title(f'Raw Data - Channel {chan_index+1} - Segment {i+1}')

            # Spectrogram for the segment
            f, t, Sxx = signal.spectrogram(segment_data, fs, nperseg=nperseg, noverlap=noverlap)
            t = t + start / fs  # Adjust time vector for the segment

            # Normalize the color range based on the data
            vmax = np.max(10 * np.log10(Sxx))
            vmin = vmax - 30
            im = axs[1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
            axs[1].set_ylabel('Frequency [Hz]')
            axs[1].set_xlabel('Time [sec]')
            axs[1].set_ylim(0, 55)
            axs[1].set_title(f'Spectrogram - Channel {chan_index+1} - Segment {i+1}')
        if action == 'show':
            plt.show()
        else:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"Segment{i+1}.png"))  # Ensure the filename is valid
            plt.close()

# Example usage:
# plot_segments(data, fs=10000, segment_length=1000, nperseg=256, noverlap=128)