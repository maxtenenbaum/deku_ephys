#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:45:21 2023

@author: maxtenenbaum
"""
import neo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch, spectrogram, cwt, filtfilt, stft, iirnotch, find_peaks
from scipy import signal
#%%

# Step 1: Import the Blackrock file
# Replace with the path to your Blackrock file
reader = neo.io.BlackrockIO(filename='data/raw/2023_10_26 NNX new settings at baseline001.ns5')
blk = reader.read_block()


#%%

#print(blk)

# Print the number of segments in the block
print("Number of segments:", len(blk.segments))

# For each segment, print information about the contained data
for i, seg in enumerate(blk.segments):
    print("Segment", i)
    print("  Number of analog signals:", len(seg.analogsignals))
    print("  Number of spike trains:", len(seg.spiketrains))
    print("  Number of events:", len(seg.events))

#%% SELECTING AND VIEWING SIGNAL PROPERTIES

# Access the first analog signal of the first segment
if len(blk.segments) > 0 and len(blk.segments[0].analogsignals) > 0:
    analog_signal = blk.segments[0].analogsignals[0]
    print(analog_signal)
    # Optionally, convert to a NumPy array and print the shape
    analog_array = analog_signal.magnitude
    print('Array shape =',analog_array.shape)
    print('Max =',analog_array.max(), '// Min =', analog_array.min(), '// Mean =', analog_array.mean())
    


#%% PLOT RAW DATA

fs = 10000
num_channels = analog_signal.shape[1]  # Number of channels

# Create time vector
time_vector = np.arange(len(analog_signal)) / fs

# Plotting each channel
plt.figure(figsize=(24, num_channels * 2))
for i in range(num_channels):
    plt.subplot(num_channels, 2, i + 1)
    plt.plot(time_vector, analog_signal[:, i].magnitude.flatten())
    plt.ylabel(f'Ch {i+1}')
    plt.xlabel('Time (s)')

plt.suptitle('Raw Data', fontsize=36)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

#%% LFP FOR ALL CHANNELS
low = 4
high = 45
def butter_bandpass_filter(signal, fs=10000):
    # Define the frequency band
    low = 4  # Hz
    high = 45  # Hz
    
    # Design the bandpass filter using butterworth filter
    nyq = 0.5 * fs  # Nyquist frequency, which is half of fs
    low = low / nyq
    high = high / nyq
    b, a = butter(1, [low, high], btype='band')

    # Apply the filter to the signal
    y = filtfilt(b, a, signal)
    
    return y


plt.figure(figsize=(24, num_channels * 2))
for i in range(num_channels):
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_signal = butter_bandpass_filter(channel_data, fs)

    plt.subplot(num_channels, 2, i + 1)
    plt.plot(time_vector, lfp_signal)
    #plt.xlim(4,6)
    plt.ylabel(f'LFP Ch {i+1}')
    plt.xlabel('Time (s)')

plt.suptitle(f'Data bandpassed between {low} Hz and {high} Hz', fontsize=36)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

#%% Looking at Standard Deviations

# Bandpass Data
for i in range(num_channels):
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_signal = butter_bandpass_filter(channel_data, fs)
    # 
# Calculate mean and standard deviation
raw_array_mean = np.mean(analog_array)
raw_array_sd = np.std(analog_array)
lfp_mean = np.mean(lfp_signal)
lfp_sd = np.std(lfp_signal) 
print("Raw Signal Mean: ",raw_array_mean)
print("Raw Signal SD: ",raw_array_sd)
print("LFP Signal Mean: ",lfp_mean,'uV')
print("LFP Signal SD: ",lfp_sd,'uV')

#%% Filtering by SD

def filter_outliers(data, num_std_dev=2):
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
#%% Full spectrum filtered

# Assuming 'analog_signal' is already defined in your workspace
fs = 10000  # Sampling frequency
num_channels = analog_signal.shape[1]  # Number of channels

# Create time vector
time_vector = np.arange(len(analog_signal)) / fs

# Plotting each channel with filtered data
plt.figure(figsize=(24, num_channels * 2))
for i in range(num_channels):
    channel_data = analog_signal[:, i].magnitude.flatten()
    filtered_data = filter_outliers(channel_data)  # Filter the channel data

    plt.subplot(num_channels, 2, i + 1)
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')

plt.suptitle('Filtered Raw Data', fontsize=36)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()
#%% LFP Raw
plt.figure(figsize=(24, num_channels * 2))
for i in range(num_channels):
    channel_data = analog_signal[:, i].magnitude.flatten()
    
    lfp_signal = butter_bandpass_filter(channel_data, fs)

    plt.subplot(num_channels, 2, i + 1)
    plt.plot(time_vector, lfp_signal)
    #plt.xlim(4,6)
    plt.ylabel(f'LFP Ch {i+1}')
    plt.xlabel('Time (s)')

plt.suptitle(f'Data bandpassed between {low} Hz and {high} Hz', fontsize=36)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

#%% LFP Filtered

plt.figure(figsize=(24, num_channels * 2))
for i in range(num_channels):
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_data = butter_bandpass_filter(channel_data, fs)  # Filter the channel data
    filtered_data = filter_outliers(lfp_data)
    plt.subplot(num_channels, 2, i + 1)
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.plot()
    plt.ylabel(f'Filtered Ch {i+1}')
    #plt.xlim(2.5,3.0)
    plt.xlabel('Time (s)')

plt.suptitle('Filtered LFP Data', fontsize=36)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()


#%% Time-frequency analysis
f, t, Zxx = signal.stft(filtered_data, 10000, nperseg=15000, noverlap=7500, window='blackman')
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.ylim(0,40)
plt.xlabel('Time [sec]')
plt.colorbar(f)
plt.show()
#%%% For all channels


plt.figure(figsize=(48, 8 * 4))  # Adjust the size as needed

for i in range(16):
    # Extract and filter the channel data
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_data = butter_bandpass_filter(channel_data, fs)
    filtered_data = filter_outliers(lfp_data)

    # Determine the column and row for plotting
    column = 1 if i < 8 else 2  # First 8 channels in column 1, rest in column 2
    row = (i % 8) * 2 + 1  # Determine the row for the filtered data

    # Plot the filtered data
    plt.subplot(16, 2, (row - 1) * 2 + column)
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')

    # Calculate and plot STFT for the filtered data
    f, t, Zxx = signal.stft(filtered_data, 10000, nperseg=10000, noverlap=5000)
    plt.subplot(16, 2, row * 2 + column)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    #plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 55)
    plt.xlabel('Time [sec]')

plt.suptitle('Filtered LFP Data and STFT Analysis', fontsize=36)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

#%% For each channels

for i in range(16):
    # Create a new figure for each channel
    plt.figure(figsize=(12, 8))  # Adjust the size as needed

    # Extract and filter the channel data
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_data = butter_bandpass_filter(channel_data, fs)
    filtered_data = filter_outliers(lfp_data)

    # Plot the filtered data
    plt.subplot(2, 1, 1)  # First subplot for filtered data
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')
    plt.title(f'Channel {i+1} - Filtered Data')

    # Calculate and plot STFT for the filtered data
    f, t, Zxx = signal.stft(filtered_data, 10000, nperseg=10000, noverlap=5000)
    plt.subplot(2, 1, 2)  # Second subplot for STFT
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 150)
    plt.xlabel('Time [sec]')

    plt.tight_layout()
    plt.show()


#%% Notch filtering and plotting
fs = 10000  # Example sampling frequency in Hz
f0 = 60    # Frequency to be removed from signal (Hz)

# Define the quality factor
Q = 60.0   # Quality factor

# Create the notch filter
b, a = signal.iirnotch(f0, Q, fs)

# Apply the filter to your data
filtered_data_notched = signal.filtfilt(b, a, filtered_data)
#%%
for i in range(16):
    # Create a new figure for each channel
    plt.figure(figsize=(12, 8))  # Adjust the size as needed

    # Extract and filter the channel data
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_data = butter_bandpass_filter(channel_data, fs)
    filtered_data = filter_outliers(lfp_data)
    filtered_data_notched = signal.filtfilt(b, a, filtered_data)
    # Plot the filtered data
    plt.subplot(2, 1, 1)  # First subplot for filtered data
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')
    plt.title(f'Channel {i+1} - Filtered Data')

    # Calculate and plot STFT for the filtered data
    f, t, Zxx = signal.stft(filtered_data_notched, 10000, nperseg=10000, noverlap=5000)
    plt.subplot(2, 1, 2)  # Second subplot for STFT
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude Notched')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 75)
    plt.xlabel('Time [sec]')

    plt.tight_layout()
    plt.show()

#%% Adding peak detection

for i in range(16):
    # Create a new figure for each channel
    plt.figure(figsize=(12, 12))  # Adjust the size as needed

    # Extract and filter the channel data
    channel_data = analog_signal[:, i].magnitude.flatten()
    lfp_data = butter_bandpass_filter(channel_data, fs)
    filtered_data = filter_outliers(lfp_data)

    # Plot the filtered data
    plt.subplot(3, 1, 1)  # First subplot for filtered data
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')
    plt.title(f'Channel {i+1} - Filtered Data')

    # Calculate and plot STFT for the filtered data
    f, t, Zxx = signal.stft(filtered_data, 10000, nperseg=10000, noverlap=5000)
    plt.subplot(3, 1, 2)  # Second subplot for STFT
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 150)
    plt.xlim(11,12)
    plt.xlabel('Time [sec]')

    # Identify peaks and overlay waveforms
    peaks, _ = find_peaks(filtered_data, threshold=0.01)  # Adjust the 'height' parameter as needed
    window_size = 50  # Adjust window size as needed

    plt.subplot(3, 1, 3)  # Third subplot for overlaid waveforms
    for peak in peaks:
        start = max(peak - window_size, 0)
        end = min(peak + window_size, len(filtered_data))
        plt.plot(filtered_data[start:end])

    plt.title('Overlaid Waveforms Around Peaks')
    plt.xlabel('Sample Number')
    plt.xlim(11,12)

    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

#%% Mean correction before filtering
# Calculate the mean across all channels for each time point
mean_across_channels = np.mean(analog_signal.magnitude, axis=1)

for i in range(16):
    # Create a new figure for each channel
    plt.figure(figsize=(12, 8))  # Adjust the size as needed

    # Extract and filter the channel data
    channel_data = analog_signal[:, i].magnitude.flatten()

    # Subtract the mean across channels from the current channel data
    channel_data_normalized = channel_data - mean_across_channels

    lfp_data = butter_bandpass_filter(channel_data_normalized, fs)
    filtered_data = filter_outliers(lfp_data)
    filtered_data_notched = signal.filtfilt(b, a, filtered_data)

    # Plot the filtered data
    plt.subplot(2, 1, 1)  # First subplot for filtered data
    plt.plot(time_vector[:len(filtered_data)], filtered_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')
    plt.title(f'Channel {i+1} - Filtered Data')

    # Calculate and plot STFT for the filtered data
    f, t, Zxx = signal.stft(filtered_data_notched, 10000, nperseg=10000, noverlap=5000)
    plt.subplot(2, 1, 2)  # Second subplot for STFT
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude Notched')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(4, 50)
    plt.xlabel('Time [sec]')

    plt.tight_layout()
    plt.show()

#%% Mean correction after filtering

# Preprocess and store filtered data for each channel
filtered_data_all_channels = []

# First pass: Filter data and store in a list
for i in range(16):
    # Extract and filter the channel data
    channel_data = analog_signal[:, i].magnitude.flatten()  # Replace with your actual data extraction method
    lfp_data = butter_bandpass_filter(channel_data, fs)
    filtered_data = filter_outliers(lfp_data)
    filtered_data_notched = signal.filtfilt(b, a, filtered_data)

    # Store the filtered and notched data
    filtered_data_all_channels.append(filtered_data_notched)

# Determine the length of the shortest array
min_length = min(len(data) for data in filtered_data_all_channels)

# Truncate all arrays to the length of the shortest array
filtered_data_all_channels = np.array([data[:min_length] for data in filtered_data_all_channels])

# Calculate the mean across all channels for each time point after filtering
mean_across_channels = np.mean(filtered_data_all_channels, axis=0)

# Normalize and plot data for each channel
for i in range(16):
    # Normalize the filtered data by subtracting the mean across channels
    normalized_data = filtered_data_all_channels[i] - mean_across_channels

    # Create a new figure for each channel
    plt.figure(figsize=(12, 8))

    # Plot the normalized filtered data
    plt.subplot(2, 1, 1)  # First subplot for normalized filtered data
    plt.plot(time_vector[:min_length], normalized_data)
    plt.ylabel(f'Filtered Ch {i+1}')
    plt.xlabel('Time (s)')
    plt.xlim(0,20)
    plt.title(f'Channel {i+1} - Normalized Filtered Data')

    # Calculate and plot STFT for the normalized filtered data
    f, t, Zxx = signal.stft(normalized_data, fs, nperseg=1000, noverlap=700)
    plt.subplot(2, 1, 2)  # Second subplot for STFT
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude Notched')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(4, 50)
    plt.xlim(0,20)
    plt.xlabel('Time [sec]')
 
    plt.tight_layout()
    plt.show()
#%%


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
        

test = power_spectral_density(analog_array)





