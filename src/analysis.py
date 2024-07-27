import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal

class Analysis:
    def __init__(self):
        self.test = 0

    def plot_stft(self, data, nperseg, noverlap, window='blackman', fs=10000):
        """
        Plots the Short-Time Fourier Transform (STFT) of the given data.
        
        Parameters:
        - data: array_like
            Time series of measurement values.
        - nperseg: int
            Length of each segment.
        - noverlap: int
            Number of points to overlap between segments.
        - window: str, optional
            Desired window to use. Options include:
            'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
            'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
            'barthann', 'kaiser' (needs beta), 'gaussian' (needs standard deviation),
            'general_gaussian' (needs power, width), 'slepian' (needs width).
            Default is 'blackman'.
        - fs: float, optional
            Sampling frequency of the `data` time series. Default is 10000.
        
        Raises:
        - ValueError: If the specified window is not one of the supported window types.
        """
        
        valid_windows = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
                         'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
                         'barthann', 'kaiser', 'gaussian', 'general_gaussian', 'slepian']
        
        if window not in valid_windows:
            raise ValueError(f"Invalid window type. Please choose from: {', '.join(valid_windows)}")
        
        f, t, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
        
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude')
        plt.show()

    def power_spectral_density(self, data, fs=10000, action='save'):
        """
        Plots the Power Spectral Density (PSD) of the given data.
        
        Parameters:
        - data: array_like
            Time series of measurement values.
        - fs: float, optional
            Sampling frequency of the `data` time series. Default is 10000.
        - action: str, optional
            Action to perform ('save' to save the plots, 'show' to display the plots). Default is 'save'.
        """
        
        for i in range(len(data[1])):
            f, S = signal.periodogram(data[i], fs, scaling='density', nfft=100)
            plt.semilogy(f, S)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
            plt.title(f"Channel {i+1} PSD")
            if action == 'show':
                plt.show()
            else:
                output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"Channel_{i+1}_PSD.png"))
                plt.close()

    def psd_all(self, data, fs=10000, title='Data PSD', action='save'):
        """
        Plots the Power Spectral Density (PSD) for all channels in the given data.
        
        Parameters:
        - data: array_like
            Time series of measurement values.
        - fs: float, optional
            Sampling frequency of the `data` time series. Default is 10000.
        - title: str, optional
            Title of the plot. Default is 'Data PSD'.
        - action: str, optional
            Action to perform ('save' to save the plots, 'show' to display the plots). Default is 'save'.
        """
        
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        num_channels = data.shape[1]
        plt.figure(figsize=(10, 6))

        for i in range(num_channels):
            f, S = signal.periodogram(data[:, i], fs, scaling='density', nfft=100)
            plt.semilogy(f, S, label=f'Ch {i+1}')

        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.ylim(0.0000001, 1000)
        plt.title(title)
        plt.legend()

        if action == 'show':
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, f"{title}.png"))
            plt.close()

    def plot_spectrogram_segments(self, data, fs, segment_length, nperseg, noverlap, action='save'):
        """
        Plots the raw data and spectrograms for segments of the given data.
        
        Parameters:
        - data: array_like
            Time series of measurement values.
        - fs: float
            Sampling frequency of the `data` time series.
        - segment_length: int
            Length of each segment.
        - nperseg: int
            Length of each segment for the spectrogram.
        - noverlap: int
            Number of points to overlap between segments.
        - action: str, optional
            Action to perform ('save' to save the plots, 'show' to display the plots). Default is 'save'.
        """
        
        num_channels = data.shape[1]
        data_length = len(data)
        num_segments = int(np.ceil(data_length / segment_length))

        for chan_index in range(num_channels):
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, data_length)
                segment_data = data[start:end, chan_index]
                time_vector = np.arange(start, end) / fs

                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(24, 8), sharex=True)

                axs[0].plot(time_vector, segment_data)
                axs[0].set_title(f'Raw Data - Channel {chan_index+1} - Segment {i+1}')

                f, t, Sxx = signal.spectrogram(segment_data, fs, nperseg=nperseg, noverlap=noverlap)
                t = t + start / fs

                vmax = np.max(10 * np.log10(Sxx))
                vmin = vmax - 30
                axs[1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
                axs[1].set_ylabel('Frequency [Hz]')
                axs[1].set_xlabel('Time [sec]')
                axs[1].set_ylim(0, 55)
                axs[1].set_title(f'Spectrogram - Channel {chan_index+1} - Segment {i+1}')
            
            if action == 'show':
                plt.show()
            else:
                output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"Channel_{chan_index+1}_Segment_{i+1}.png"))
                plt.close()

# Example usage:
# analysis = Analysis()
# analysis.plot_stft(data, nperseg=256, noverlap=128)
# analysis.power_spectral_density(data)
# analysis.psd_all(data)
# analysis.plot_spectrogram_segments(data, fs=10000, segment_length=1000, nperseg=256, noverlap=128)
