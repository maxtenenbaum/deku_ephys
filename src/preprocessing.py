import neo
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class DataLoading:
    def __init__(self, filepath):
        self.filepath = filepath

    def blkrck_to_numpy(self):
        """
        Converts Blackrock file data to a NumPy array.

        Returns:
        - data_array: NumPy array of the analog signal data.
        """
        reader = neo.io.BlackrockIO(filename=self.filepath)
        blk = reader.read_block()
        print("Number of segments:", len(blk.segments))
        for i, seg in enumerate(blk.segments):
            print("Segment", i)
            print("  Number of analog signals:", len(seg.analogsignals))
            print("  Number of spike trains:", len(seg.spiketrains))
            print("  Number of events:", len(seg.events))
        if len(blk.segments) > 0 and len(blk.segments[0].analogsignals) > 0:
            analog_signal = blk.segments[0].analogsignals[0]
            print(analog_signal)
            
            data_array = analog_signal.magnitude
            print('Array shape =', data_array.shape)
            print('Max =', data_array.max(), '// Min =', data_array.min(), '// Mean =', data_array.mean())
            
            return data_array
        else:
            return None


class Preprocessing:
    def __init__(self):
        pass
    
    def butter_bandpass_filter(self, signal_data, low, high, fs=10000):
        """
        Applies a Butterworth bandpass filter to the given signal data.
        
        Parameters:
        - signal_data: array_like
            The input signal data.
        - low: float
            The low cutoff frequency.
        - high: float
            The high cutoff frequency.
        - fs: float, optional
            The sampling frequency of the signal. Default is 10000.
        
        Returns:
        - filtered_data: array_like
            The bandpass filtered signal data.
        """
        nyq = 0.5 * fs
        low = low / nyq
        high = high / nyq
        b, a = butter(1, [low, high], btype='band')
        filtered_data = filtfilt(b, a, signal_data)
        return filtered_data

    def filter_outliers(self, data, num_std_dev=3):
        """
        Filters out data points that are more than num_std_dev standard deviations from the mean.
        
        Parameters:
        - data: array_like
            NumPy array of data points.
        - num_std_dev: int, optional
            Number of standard deviations for the cutoff. Default is 3.
        
        Returns:
        - filtered_data: array_like
            Filtered NumPy array.
        """
        mean = np.mean(data)
        std_dev = np.std(data)
        lower_bound = mean - num_std_dev * std_dev
        upper_bound = mean + num_std_dev * std_dev
        return data[(data >= lower_bound) & (data <= upper_bound)]


    def notch_filter(self, data, fs=10000, f0=60, Q=30):
        """
        Applies a notch filter to remove a specific frequency from the data.
        
        Parameters:
        - data: array_like
            The input signal data.
        - fs: float, optional
            The sampling frequency of the signal. Default is 10000.
        - f0: float, optional
            The frequency to remove. Default is 60.
        - Q: float, optional
            The quality factor. Default is 30.
        
        Returns:
        - filtered_data: array_like
            The notch filtered signal data.
        """
        b, a = iirnotch(f0, Q, fs)
        filtered_data = filtfilt(b, a, data)
        return filtered_data