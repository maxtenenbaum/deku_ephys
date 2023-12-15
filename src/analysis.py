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