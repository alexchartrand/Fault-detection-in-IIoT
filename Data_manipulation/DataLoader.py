import numpy as np
from os import path, listdir
from constant import *
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def getAllcsv(csvFolder):
    return [path.join(csvFolder, f) for f in listdir(csvFolder)
            if path.isfile(path.join(csvFolder, f)) and f.lower().endswith('.csv')]

class MotorFaultDataset(Dataset):
    """Motro fault dataset"""

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

    def _generateFileName(self, index):

        fileNames = []
        for i in index:
            fileNames.append("{:05d}.csv".format(i))

        return fileNames

    def getMotorsData(self):
        return self.motors

    def __len__(self):
        return self.motors.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        motor_name = path.join(self.root_dir,
                             self.motors.iloc[idx]["file_name"])

        data = np.genfromtxt(motor_name, delimiter=",", skip_header=1)
        motor = self.motors.loc[idx, self.motors.columns != "file_name"]
        motor = np.array([motor])
        motor = motor.astype('float')  #.reshape(-1, 3)
        sample = {'data': data, 'fault': motor}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, fault = sample['data'], sample['fault']

        return {'data': torch.from_numpy(data),
                'fault': torch.from_numpy(fault)}

if __name__ == "__main__":
    motorDataset = MotorFaultDataset(path.join(SIMULATION_MOTOR_FOLDER, "result.csv"), SIMULATION_MOTOR_FOLDER)

    dataloader = DataLoader(motorDataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['data'].size(),
              sample_batched['fault'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            break

    print("done")

