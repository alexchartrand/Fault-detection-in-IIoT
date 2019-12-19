import numpy as np
from os import path, listdir
from constant import *
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import DataTransform

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

    def __len__(self):
        return self.motors.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        motor_name = path.join(self.root_dir,
                             self.motors.iloc[idx]["file_name"])

        data = np.genfromtxt(motor_name, delimiter=",", skip_header=1)
        data = np.transpose(data)
        if not USE_TIME:
            data = np.delete(data, TIME_IDX, axis=0) # We remove the time variable for simplicity

        motor = self.motors.loc[idx, self.motors.columns != "file_name"]
        motor = np.array(motor)
        motor = motor.astype('float')  #.reshape(-1, 3)
        sample = {'data': data, 'fault': motor, 'minMax': self._motorMinMax[motor[0]]}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    motorDataset = MotorFaultDataset(csv_file=path.join(SIMULATION_MOTOR_FOLDER, "result.csv"),
                                     root_dir=SIMULATION_MOTOR_FOLDER,
                                     transform=transforms.Compose([DataTransform.To3DTimeSeries(), DataTransform.ToTensor()]) )

    dataloader = DataLoader(motorDataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['data'].size(),
              sample_batched['fault'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            break

    print("done")

