import numpy as np
import csv
from os import path, listdir

DATA_FOLDER = "D:\\OneDrive\\Documents\\ETS\\Memoire\\Simulation models\\Data"
NP_SAVE_FOLDER = "D:\\OneDrive\\Documents\\ETS\\Memoire\\IoT\\data"
NP_SIM_FILE = path.join(NP_SAVE_FOLDER, "sim.npy")
NP_PARAM_FILE = path.join(NP_SAVE_FOLDER, "param.npy")

def createDataFiles(dataFolder):
    dataFiles = [path.join(dataFolder, f) for f in listdir(dataFolder) if path.isfile(path.join(dataFolder, f))]
    param = None
    sim = []
    for f in dataFiles:
        dNp = np.genfromtxt(f, delimiter=",", skip_header=1)

        if "param" in path.basename(f):
            param = dNp
        else:
            sim.append(dNp)

    np.save(NP_SIM_FILE, sim)
    np.save(NP_PARAM_FILE, param)

def getDataHeaders(dataFolder):
    dataFiles = [path.join(dataFolder, f) for f in listdir(dataFolder) if path.isfile(path.join(dataFolder, f))]

def getParamData():
    return np.load(NP_PARAM_FILE)

def getSimData():
    return np.load(NP_SIM_FILE)


if __name__ == "__main__":
    createDataFiles(DATA_FOLDER)

    print("Done")
