import numpy as np
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


def power_spectral_density(data, fs=10000):
    for i in range(len(data[1])):
        (f, S) = signal.periodogram(data[i], fs, scaling='density', nfft=100)
        plt.semilogy(f, S)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        #plt.xlim(2,2000)
        #plt.ylim(10*1,10**9)
        plt.title(f"Channel {i+1} PSD")
        plt.show()
        