from DataLoader import getSimData

class SimData:
    COL_NAME = ["time", "currant", "torque", "speed"]

    def __init__(self, simName, dataArray):
        self._simName = simName
        self._data = dataArray
        self._fault = None

    def __getitem__(self, item):
        return self._data[item,:]

    def _getColIndex(self, colName):
        return self.COL_NAME.index(colName)

    def getCol(self, colName):
        index = self._getColIndex(colName)

        if index >= 0:
            return self._data[:,index]
        else:
            return []

    def name(self):
        return self._simName

def getSims():
    sim = getSimData()
    simNo = 1
    simArray = []
    for i in range(sim.shape[0]):
        simArray.append(SimData(str(simNo), sim[i,:,:]))
        simNo += 1

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