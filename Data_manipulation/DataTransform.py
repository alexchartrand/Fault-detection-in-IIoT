import torch
import numpy as np
from constant import  *
import Data_manipulation.DataLoader as DataLoader
from Plotting.PlotSetup import *

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, fault, minMax = sample['data'], sample['fault'], sample['minMax']
        sample['data'] = torch.from_numpy(data).float()
        sample['fault'] = torch.from_numpy(fault).float()
        sample['minMax'] = torch.from_numpy(minMax).float()
        return sample

class ToNumpy(object):
    """Convert torch.tensor to ndArray."""

    def __call__(self, sample):
        data, fault, minMax = sample['data'], sample['fault'], sample['minMax']
        sample['data'] = data.cpu().detach().numpy()
        sample['fault'] = fault.cpu().detach().numpy()
        sample['minMax'] = minMax.cpu().detach().numpy()
        return sample

class To3DTimeSeries(object):
    """Convert to 3D timeseries"""

    def __call__(self, sample):
        data = sample['data']
        sample['data'] = torch.unsqueeze(data, 2)
        return sample

class Derivative(object):
    """First order derivation of data"""

    def __call__(self, sample):
        data = sample['data']
        x_diff = data[:,1:] - data[:,:-1]
        sample['data'] = x_diff
        return sample

class Frequency(object):
    """FFT transform"""

    def __call__(self, sample):
        data = sample['data']
        fft = np.fft.rfftn(data)
        sample['data'] = fft
        return sample

def plotFFT(data):
    fig, axs = plt.subplots(3, sharex='col')
    N = 2501
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    axs[0].plot(xf, 2.0/N * np.abs(data[CURRANT_IDX][:N//2]))
    setupAx(axs[0])
    axs[0].set_title('Courant')
    axs[0].set(ylabel='A')
    axs[1].plot(xf, 2.0/N * np.abs(data[VOLTAGE_IDX][:N//2]))
    setupAx(axs[1])
    axs[1].set_title('Tension')
    axs[1].set(ylabel='V')

    axs[2].plot(xf, 2.0/N * np.abs(data[SPEED_IDX][:N//2]))
    setupAx(axs[2])
    axs[2].set_title('Vitesse')
    axs[2].set(ylabel='rad/s')

    for ax in axs.flat:
        ax.set(xlabel='Frequency')
        ax.grid(True)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout()
    plt.show()
    return fig, axs

if __name__ == "__main__":
   motorDataset = DataLoader.MotorFaultDataset(csv_file=path.join(SIMULATION_MOTOR_FOLDER,"simulation", "result.csv"),
                                     root_dir=path.join(SIMULATION_MOTOR_FOLDER,"simulation"))
   freq = Frequency()
   sample = freq(motorDataset[2005])
   plotFFT(sample["data"])