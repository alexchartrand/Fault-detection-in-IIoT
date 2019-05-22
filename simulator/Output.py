

class OutputBase:
    """ All class used in the simulator as output must
        inherits OutputBase"""

    def __init__(self):
        pass

    def write(self, time, data):
        raise NotImplementedError

class OutputConsol(OutputBase):

    def __init__(self):
        super().__init__()

    def write(self, time, data):
        print("{:.4f}\t".format(time), end='')

        for v in data:
            print("{:.4f}\t".format(v), end='')

        print('', flush=True)
