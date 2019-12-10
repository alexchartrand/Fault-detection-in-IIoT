# source: https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
# fft source: https://stackoverflow.com/questions/36968418/python-designing-a-time-series-filter-after-fourier-analysis

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



# make up a noisy signal
dt=0.01
t= np.arange(0,5,dt)
f1,f2= 5, 20  #Hz
n=t.size
s0=  0.2*np.sin(2*np.pi*f1*t)+ 0.15 * np.sin(2*np.pi*f2*t)
sr= np.random.rand(np.size(t))
s=s0+sr

#fft
s-= s.mean()  # remove DC (spectrum easier to look at)
fr=np.fft.fftfreq(n,dt)  # a nice helper function to get the frequencies
fou=np.fft.fft(s)

#make up a narrow bandpass with a Gaussian
df=0.1
gpl= np.exp(- ((fr-f1)/(2*df))**2)+ np.exp(- ((fr-f2)/(2*df))**2)  # pos. frequencies
gmn= np.exp(- ((fr+f1)/(2*df))**2)+ np.exp(- ((fr+f2)/(2*df))**2)  # neg. frequencies
g=gpl+gmn
filt=fou*g  #filtered spectrum = spectrum * bandpass

#ifft
s2=np.fft.ifft(filt)

plt.figure(figsize=(12,8))

plt.subplot(511)
plt.plot(t,s0)
plt.title('data w/o noise')

plt.subplot(512)
plt.plot(t,s)
plt.title('data w/ noise')

plt.subplot(513)
plt.plot(np.fft.fftshift(fr) ,np.fft.fftshift(np.abs(fou) )  )
plt.title('spectrum of noisy data')

plt.subplot(514)
plt.plot(fr,g*50, 'r')
plt.plot(fr,np.abs(filt))
plt.title('filter (red)  + filtered spectrum')

plt.subplot(515)
plt.plot(t,np.real(s2))
plt.title('filtered time data')
