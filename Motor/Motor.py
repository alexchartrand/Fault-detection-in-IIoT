from Data_manipulation.ParamData import getParam
from Data_manipulation.SimData import getSims
from copy import deepcopy
from Data_manipulation.DataVisualization import plotSimulation
import Data_manipulation.Filtering as filt

class Motor:

    def __init__(self, simData, paramData):
        self._simData = simData
        self._paramData = paramData

        self._fault = self._paramData.getFault()
        timeArray = self._simData.getCol("time")
        self.SAMPLING_TIME = timeArray[1] - timeArray[0]

        self._steadyStateData = self._createSteadyState()

    def _createSteadyState(self):
        steadyStateStart = 0.5
        steadyStateIndex = int(steadyStateStart / self.SAMPLING_TIME)
        d = deepcopy(self._simData)
        d.truncateData(steadyStateIndex)
        return d

    def applyLowPassFilter(self, cutoof_hz, order):
        cutoff_hz = 100
        order = 3
        freq = 1.0 / self.SAMPLING_TIME

        self._steadyStateData = self._createSteadyState()

    def fault(self):
        return self._fault

    def simData(self):
        return self._simData

    def paramData(self):
        return self._paramData

    def steadyState(self):
        return self._steadyStateData

    def show(self):
        plotSimulation(self._simData, self._fault)

    def showSteadyState(self):
        plotSimulation(self._steadyStateData, self._fault)

def getMotorsData():
    simRes = getSims()
    param = getParam()
    motorArray = []
    for i in range(len(simRes)):
        motorArray.append(Motor(simRes[i], param[i]))

    return  motorArray

def filterData(motor):
    cutoff_hz = 100
    order = 3
    freq = 1.0/motor.SAMPLING_TIME
    time = motor.simData().getCol("time")
    data = motor.simData().getCol("currant")

    filt.showFilter(cutoff_hz, freq, order)
    filt.showFilteredData(time, data, cutoff_hz, freq, order)

if __name__ == "__main__":
    motorArray = getMotorsData()

    filterData(motorArray[1])