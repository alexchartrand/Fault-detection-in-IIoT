import numpy as np
import torch
from torchvision import transforms
import DataTransform
from PlotSetup import *
from constant import  *
import DataLoader

class NoramalizeMinMax(object):
    """Normalize data using min max technique"""

    def __init__(self, cliping=False):
        self._cliping = cliping

    def __call__(self, sample):
        data, minMax = sample['data'], sample['minMax']
        low = -1 * torch.ones((data.shape[0], 1))
        high = 1 * torch.ones((data.shape[0], 1))
        min = minMax[:,0].unsqueeze(1)
        max = minMax[:, 1].unsqueeze(1)
        if self._cliping:
            data = torch.max(torch.min(data, max), min)

        normalizedData = (high-low)*((data-min)/(max-min)) + low

        sample['data'] = normalizedData
        return sample

class NoramalizeMinMaxFixWindow(object):
    """Normalize data using min max technique"""

    def __init__(self, sampleWindowSize, cliping=False):
        self._cliping = cliping
        self._sampleWindowSize = sampleWindowSize

    def __call__(self, sample):
        data = sample['data']
        low = -1 * torch.ones((data.shape[0], 1))
        high = 1 * torch.ones((data.shape[0], 1))

        maxValues = torch.max(data[:,0:self._sampleWindowSize], 1, True)[0].float()
        minValues = torch.min(data[:,0:self._sampleWindowSize], 1, True)[0].float()

        if self._cliping:
            data = torch.max(torch.min(data, maxValues), minValues)

        normalizedData = (high-low)*((data-minValues)/(maxValues-minValues)) + low
        sample['data'] = normalizedData
        return sample

class NoramalizeZ(object):
    """Normalize data using z-score technique"""
    def __init__(self, sampleWindowSize):
        self._sampleWindowSize = sampleWindowSize

    def __call__(self, sample):
        data = sample['data']

        means  = torch.mean(data[:,0:self._sampleWindowSize], 1, keepdim=True)
        stds = torch.std(data[:,0:self._sampleWindowSize], 1, keepdim=True)

        normalizedData = (data - means) / stds
        sample['data'] = normalizedData
        return sample


class NoramalizeSlidingWindow(object):
    """Normalize data using sliding windows technique"""

    def __init__(self, windowSize):
        self._windowSize = windowSize
    def __call__(self, sample):
        data = sample['data']

        #low = -1 * torch.ones((data.shape[0], 1))
        #high = 1 * torch.ones((data.shape[0], 1))

        normalizedData = torch.zeros((data.shape[0],self._windowSize-1))
        for idx in range(0, data.shape[1] - self._windowSize+1):
            window = data[:,idx:idx+self._windowSize]
            newDaTA = window[:, -1:]

            #maxValues = torch.max(window, 1, True)[0].float()
            #minValues = torch.min(window, 1, True)[0].float()
            #windowNormalized = (high-low)*((newDaTA-minValues)/(maxValues-minValues)) + low

            means = torch.mean(window, 1, keepdim=True)
            stds = torch.std(window, 1, keepdim=True)

            windowNormalized = (newDaTA - means) / stds
            normalizedData = torch.cat((normalizedData, windowNormalized), dim=1)

        sample['data'] = normalizedData
        return sample


class NoramalizeAN(object):
    """Normalize data using adaptative normalization technique
        Paper: Adaptive Normalization: A Novel Data Normalization Approach for Non-Stationary Time Series"""

    def __init__(self, windowSize):
        self._windowSize = windowSize

    def __call__(self, sample):
        data = sample['data']

        return sample

class EmptyNormalization(object):
    def __call__(self, sample):
        return sample

normalizationTransform = {
                          "Vrai Min/Max": transforms.Compose([DataTransform.ToTensor(), NoramalizeMinMax(), DataTransform.ToNumpy()])}

def plotNormalizationTechnique(motorData, indexToPlot):
    fig, ax = plt.subplots()

    alphaVal = 0.9
    linethick = 2.5

    x = np.arange(0, motorData["data"].shape[1], 1)
    i=0
    for tName, transform in normalizationTransform.items():
        m = transform(motorData)["data"]
        ax.plot(x, # second to mS
                m[indexToPlot,:],
                color=colourWheel[i % len(colourWheel)],
                linestyle='-',
                lw=linethick,
                label="{}".format(tName),
                alpha=alphaVal)
        i += 1

    setupAx(ax)
    ax.legend(frameon=False, loc='upper left', ncol=1, handlelength=4)
    plt.grid()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    motorDataset = DataLoader.MotorFaultDataset(csv_file=path.join(SIMULATION_MOTOR_FOLDER, "result.csv"),
                                     root_dir=SIMULATION_MOTOR_FOLDER)
    testData = motorDataset[1]
    plotNormalizationTechnique(testData, SPEED_IDX)