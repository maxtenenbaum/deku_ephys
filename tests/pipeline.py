from src import preprocessing, visualization, analysis
import matplotlib
import numpy as np
from scipy.signal import butter
matplotlib.use('module://matplotlib_inline.backend_inline')

raw_europa_data = preprocessing.blkrck_to_numpy('data/raw/121323T1E1_Europa16_E6_30kHz003.ns5')
bandpassed_europa = preprocessing.butter_bandpass_filter(raw_europa_data, 4, 200)

#bandpass_plot = visualization.plot_data(bandpassed_europa, title='Europa Bandpass')
#raw_europa_plot = visualization.plot_data(raw_europa_data, title='Raw Data')
#andpass_psd = analysis.psd_all(bandpassed_europa, title='Bandpass Europa PSD')
nperseg = 10000
noverlap = 200

raw_spectrogram = analysis.plot_spectrogram_segments(raw_europa_data, 10000, 15, nperseg, noverlap)