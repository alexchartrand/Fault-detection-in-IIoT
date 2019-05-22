from Input import InputBase
from ctypes import *

MODEL_DLL_PATH = "D:\OneDrive\Documents\ETS\Memoire\IoT\simulator\SimpleMotor_sim_win64.dll"

class ExtU(Structure):
    _fields_  = [("V", c_double)]

class ExtY(Structure):
    _fields_  = [("Motor_speed", c_double),
                 ("Motor_current", c_double)]

class SimpleMotor(InputBase):

    def __init__(self, T):
        super().__init__()
        self._motor = cdll.LoadLibrary(MODEL_DLL_PATH)
        self._motor.SimpleMotor_sim_initialize()
        self._T = T
        self._tick = 0
        self._out = ExtY.in_dll(self._motor, "rtY")
        self._in = ExtU.in_dll(self._motor, "rtU")

    def setVoltage(self, v):
        self._in.V = v

    def getNextTick(self):
        nextTick = self._tick
        self._tick += self._T
        return nextTick

    def getNextData(self, tick):
        self._motor.SimpleMotor_sim_step()
        return [self._out.Motor_speed, self._out.Motor_current]

    def __del__(self):
        if self._motor:
            self._motor.SimpleMotor_sim_terminate()