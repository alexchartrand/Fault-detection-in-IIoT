# source: https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def createButterLowpassFilter(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def applyButterLowpassFilter(data, cutoff, fs, order):
    b, a = createButterLowpassFilter(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

def showFilter(cutoff, fs, order):

    b, a = createButterLowpassFilter(cutoff, fs, order)
    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid(True)
    plt.show()

def showFilteredData(time, data, cutoff, fs, order):
    # Filter the data, and plot both the original and filtered signals.
    y = applyButterLowpassFilter(data, cutoff, fs, order)

    plt.plot(time, data, 'b-', label='data')
    plt.plot(time, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid(True)
    plt.legend()
    plt.show()
