import numpy as np


class InputBase:
    """ All class used in the simulator as input must
    inherits InputBase"""

    def __init__(self, use_fix_tick=False):
        self._use_fix_tick = use_fix_tick

    def getNextTick(self):
        if self._use_fix_tick:
            pass
        else:
            raise NotImplementedError

    def getNextData(self, tick):
        raise NotImplementedError

class InputCSV(InputBase):

    def __init__(self, csvFilePath):
        super().__init__()
        csv = np.genfromtxt(csvFilePath, dtype=float, delimiter=',', skip_header=1)
        self._tickIndex = 0
        self._tickArray = csv[:, 0]

        self._dataIndex = 0
        self._dataArray = csv[:, 1:]

    def getNextTick(self):
        nextTick = self._tickArray[self._tickIndex]
        self._tickIndex += 1
        return nextTick

    def getNextData(self, tick):
        nextData = self._dataArray[self._dataIndex]
        self._dataIndex += 1
        return nextData
