import numpy as np
from os import path, listdir
from constant import *
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def getAllcsv(csvFolder):
    return [path.join(csvFolder, f) for f in listdir(csvFolder)
            if path.isfile(path.join(csvFolder, f)) and f.lower().endswith('.csv')]

class MotorFaultDataset(Dataset):
    """Motor fault dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the motors time series.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.motors = pd.read_csv(csv_file)
        self.motors["id"] = self.motors.index
        self.motors["file_name"] = self._generateFileName(self.motors.index)
        self.root_dir = root_dir
        self.transform = transform
        self._motorMinMax = self._loadMotorsMinMax()

        # caching section
        shared_array_base = [None] * self.motors.shape[0]
        self.shared_array = shared_array_base
        self.use_cache = False

    @staticmethod
    def _generateFileName(index):

        fileNames = []
        for i in index:
            fileNames.append("{:05d}.csv".format(i))

        return fileNames

    def _loadMotorsMinMax(self):
        regex = r"^motor(\d+)\.csv$"

        motorsFile = [path.join(self.root_dir, f) for f in listdir(self.root_dir)
                      if re.search(regex, f, re.IGNORECASE)]
        motorArray = dict()
        for motor in motorsFile:
            f = path.basename(motor)
            matches = re.search(regex, f, re.IGNORECASE)
            motorArray[int(matches.group(1))] = pd.read_csv(motor).to_numpy()
        return motorArray

    def getMotorsData(self):
        return self.motors

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def getPlotableData(self, idx):
        data = self[idx]['data']
        if torch.is_tensor(data):
            data = data.cpu().detach().numpy()
        time_array = np.arange(0, data.shape[1] * T, T)
        return np.vstack((data, time_array))

    def __len__(self):
        return self.motors.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.use_cache and self.shared_array[idx]:
            return self.shared_array[idx]

        sample = self.getSample(idx)

        if self.use_cache:
            self.shared_array[idx] = sample

        return sample

    def getSample(self, idx):

        motor_name = path.join(self.root_dir,
                             self.motors.iloc[idx]["file_name"])

        data = np.genfromtxt(motor_name, delimiter=",", skip_header=1)
        data = np.transpose(data)
        data = np.delete(data, 0, axis=0) # We remove the time variable for simplicity

        motor = self.motors.loc[idx, self.motors.columns != "file_name"]
        motor = np.array(motor)
        motor = motor.astype('float')  #.reshape(-1, 3)
        sample = {'data': data, 'fault': motor, 'minMax': self._motorMinMax[motor[0]]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def createTrainDataLoader(dataFolder, batch_size, transform, use_cache):
    motor_train = MotorFaultDataset(csv_file=path.join(dataFolder, "simulation", "result.csv"),
                                         root_dir=path.join(dataFolder, "simulation"),
                                         transform=transform)
    motor_train.set_use_cache(use_cache)

    # Creating data indices for training and validation splits:
    len_data = len(motor_train)
    indices = list(range(len_data))
    #id_split = int(0.85 * len_data)
    #np.random.shuffle(indices)
    #train_indices, valid_indices = indices[:id_split], indices[id_split:]

    # Creating PT data samplers and loaders:
    number_of_data_per_motor = 2500
    train_sampler = SubsetRandomSampler(indices[:2*number_of_data_per_motor-1])
    valid_sampler = SubsetRandomSampler(indices[2*number_of_data_per_motor:])

    num_worker = 4
    if [use_cache]:
        num_worker = 0

    train_loader = DataLoader(motor_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker)
    valid_loader = DataLoader(motor_train, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker)
    return train_loader, valid_loader

def createTestDataLoader(dataFolder, batch_size, transform):

    motor_test = MotorFaultDataset(csv_file=path.join(dataFolder, "simulation", "test", "result.csv"),
                                         root_dir=path.join(dataFolder, "simulation", "test"),
                                         transform=transform)

    test_loader = DataLoader(motor_test, batch_size=batch_size, num_workers=4)
    return test_loader

