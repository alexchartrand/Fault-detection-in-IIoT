from DataLoader import getParamData as dat

COL_NAME = ["sim_no", "Kt", "Kt_delta", "Ke", "Ke_delta", "Ra", "Ra_delta", "Rt", "Rt_delta", "L", "J", "Bm"]
FAULT_ARRAY = [COL_NAME.index(i) for i in COL_NAME if "delta" in i]

class ParamData:

    def __init__(self, simName, dataArray):
        self._simName = simName
        self._data = dataArray

    def __getitem__(self, item):
        return self._data[item]

    def _getColIndex(self, colName):
        return COL_NAME.index(colName)

    def getCol(self, colName):
        index = self._getColIndex(colName)

        if index >= 0:
            return self._data[index]
        else:
            return []

    def getValue(self, colName):
        index = self._getColIndex(colName)

        if index >= 0:
            return self._data[index]
        else:
            return None

    def name(self):
        return self._simName

    def getFault(self):

        for i in FAULT_ARRAY:
            if self._data[i] != 0:
                return COL_NAME[i].replace("_delta", "")

        return "no_fault"

    def __deepcopy__(self, memodict={}):
        return ParamData(self.name(), self._data.copy())


def getParam():
    param = dat()
    paramArray = []
    for i in range(param.shape[0]):
        paramArray.append(ParamData(str(i), param[i]))

    return paramArray



if __name__ == "__main__":
    p = getParam()
