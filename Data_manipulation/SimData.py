from DataLoader import getSimData as dat

COL_NAME = ["time", "currant", "torque", "speed"]

class SimData:

    def __init__(self, simName, dataArray):
        self._simName = simName
        self._data = dataArray

    def __getitem__(self, item):
        return self._data[item,:]

    def _getColIndex(self, colName):
        return COL_NAME.index(colName)

    def getCol(self, colName):
        index = self._getColIndex(colName)

        if index >= 0:
            return self._data[:,index]
        else:
            return []

    def data(self):
        return self._data

    def truncateData(self, index):
        self._data = self._data[index:,:]

    def name(self):
        return self._simName

    def __deepcopy__(self, memodict={}):
        return type(self)(self.name(), self._data.copy())


def getSims():
    sim = dat()
    simArray = []
    for i in range(sim.shape[0]):
        simArray.append(SimData(str(i), sim[i,:,:]))

    return  simArray


if __name__ == "__main__":
    # sim = getSimData()
    # h = ["time", "currant", "torque", "speed"]
    # simNo = 1
    # s = SimData(str(simNo), sim[simNo,:,:])
    #
    # print(s[5])
    # print(s.getCol("time"))
    res = getSims()