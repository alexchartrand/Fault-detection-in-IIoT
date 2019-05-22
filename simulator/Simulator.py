from Input import InputCSV
from Output import OutputConsol
from SimpleMotor import SimpleMotor
import time

T=0.1

class Simulator:

    def __init__(self, input, output):
        self._isRunning = False
        self._input = input
        self._output = output

    def run(self):
        self._isRunning = True

        while self._isRunning:

            t = self._input.getNextTick()
            d = self._input.getNextData(t)
            self._output.write(t, d)
            time.sleep(T)

    def stop(self):
        self._isRunning = False


if __name__ == "__main__":

    dataPath = "D:\OneDrive\Documents\ETS\Mémoire\IoT\data\SimpleMotor\data.csv"
    motor = SimpleMotor(T)
    motor.setVoltage(100)
    s = Simulator(motor, OutputConsol())
    s.run()