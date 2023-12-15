import neo
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
def blkrck_to_numpy(filepath):
    reader = neo.io.BlackrockIO(filename=f'{filepath}')
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
        
        # Convert to a NumPy array
        data_array = analog_signal.magnitude
        print('Array shape =', data_array.shape)
        print('Max =', data_array.max(), '// Min =', data_array.min(), '// Mean =', data_array.mean())
        
        return data_array
    else:
        return None

# Example usage 'data' saved as NumPy array:
"""
data = blkrck_to_numpy("data/raw/2023_11_20 RefScrewLocation1001.ns6")
"""


def butter_bandpass_filter(signal_data, low, high, fs=10000):
    # Design the bandpass filter using butterworth filter
    nyq = 0.5 * fs  # Nyquist frequency, which is half of fs
    low = low / nyq
    high = high / nyq
    b, a = butter(1, [low, high], btype='band')

    # Apply the filter to the signal
    filtered_data = filtfilt(b, a, signal_data)
    
    return filtered_data

# Example usage
"""
fitered_data = butter_bandpass_filter(data)
"""

def filter_outliers(data, num_std_dev=3):
    """
    Filters out data points that are more than num_std_dev standard deviations from the mean.
    
    :param data: NumPy array of data points
    :param num_std_dev: Number of standard deviations for the cutoff
    :return: Filtered NumPy array
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - num_std_dev * std_dev
    upper_bound = mean + num_std_dev * std_dev
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Example usage
"""
filtered_data = filter_outliers(data)
"""

