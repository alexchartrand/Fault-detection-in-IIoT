import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoader import getDataFile
sns.set()

def plotSimulation(simNDarray):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Simulation result")
    time = simNDarray[:,0]

    # Currant
    axs[0, 0].plot(time, simNDarray[:,1])
    axs[0, 0].set_title('Currant')
    axs[0, 0].set(xlabel='Time (s)', ylabel='Currant (A)')

    # Torque
    axs[0, 1].plot(time, simNDarray[:,2], 'tab:orange')
    axs[0, 1].set_title('Torque')
    axs[0, 1].set(xlabel='Time (s)', ylabel='Torque (N/m)')

    # Speed
    axs[1, 0].plot(time, simNDarray[:,3], 'tab:green')
    axs[1, 0].set_title('Speed')
    axs[1, 0].set(xlabel='Time (s)', ylabel='Speed (m/s)')

    # Param
    #axs[1, 1].plot(time, -y, 'tab:red')
    #axs[1, 1].set_title('Param')

    plt.tight_layout()


if __name__ == "__main__":
    param, sim = getDataFile()

    plotSimulation(sim[0])