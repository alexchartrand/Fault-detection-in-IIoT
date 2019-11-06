import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from ParamData import getParam
from SimData import getSims
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def interPoalateData(x, y):
    xnew = np.linspace(x.min(), x.max(), 1000)  # 300 represents number of points to make between T.min and T.max

    spl = make_interp_spline(x, y, k=3)  # BSpline object
    ynew = spl(xnew)

    return xnew, ynew

def plotSimulation(simData, fault):
    fig, axs = plt.subplots(3, sharex='col') #gridspec_kw={'hspace': 1}
    fig.suptitle("Simulation result with fault on: {}".format(fault))
    time = simData.getCol("time")

    # Currant
    newTime, newDat = interPoalateData(time, simData.getCol("currant"))
    axs[0].plot(newTime, newDat)
    axs[0].set_title('Currant')
    axs[0].set(ylabel='Currant (A)')

    # Torque
    newTime, newDat = interPoalateData(time, simData.getCol("torque"))
    axs[1].plot(newTime, newDat, 'tab:orange')
    axs[1].set_title('Torque')
    axs[1].set(ylabel='Torque (N/m)')

    # Speed
    newTime, newDat = interPoalateData(time, simData.getCol("speed"))
    axs[2].plot(newTime, newDat, 'tab:green')
    axs[2].set_title('Speed')
    axs[2].set(ylabel='Speed (rad/s)')

    for ax in axs.flat:
        ax.set(xlabel='Time (s)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    param = getParam()
    sim = getSims()
    simNo = 10
    plotSimulation(sim[simNo], param.getFault(simNo))