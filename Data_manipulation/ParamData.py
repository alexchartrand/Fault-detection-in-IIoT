from DataLoader import getParamData


class ParamData:
    COL_NAME = ["sim_no", "Kt", "Kt_delta", "Ke", "Ke_delta", "Ra", "Ra_delta", "Rt", "Rt_delta", "L", "J", "Bm"]

    def __init__(self, dataArray):
        self._data = dataArray
        self._faultArray = [self.COL_NAME.index(i) for i in self.COL_NAME if "delta" in i]

    def __getitem__(self, item):
        return self._data[item-1,:]

    def _getColIndex(self, colName):
        return self.COL_NAME.index(colName)

    def getCol(self, colName):
        index = self._getColIndex(colName)

        if index >= 0:
            return self._data[:,index]
        else:
            return []

    def getValue(self, sim_no, colName):
        simParam = self[sim_no]
        index = self._getColIndex(colName)

        if index >= 0:
            return simParam[index]
        else:
            return None

    def getFault(self, sim_no):
        simParam = self[sim_no]

        for i in self._faultArray:
            if simParam[i] != 0:
                return self.COL_NAME[i].replace("_delta", "")

        return "no_fault"


def getParam():
    param = getParamData()
    return ParamData(param)


if __name__ == "__main__":
    param = getParamData()

    p = ParamData(param)
    print(p.getFault(2))
