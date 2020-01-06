from os import path
from constant import  *
from Data_manipulation.DataLoader import MotorFaultDataset
from Plotting.PlotSetup import *


def plotMotors(motors, yValueName):
    fig, ax = plt.subplots()

    alphaVal = 0.75
    linethick = 2.5
    timeLimit = 1000

    for i in range(len(motors)):
        m = motors[i]
        m = m[:][0:timeLimit]
        ax.plot(m["time"]*1000, # second to mS
                m[yValueName],
                color=colourWheel[i % len(colourWheel)],
                linestyle='-',
                dashes=dashesStyles[i % len(dashesStyles)],
                lw=linethick,
                label="Motor {}".format(i+1),
                alpha=alphaVal)

    setupAx(ax)
    ax.legend(frameon=False, loc='upper right', ncol=1, handlelength=4)
    plt.grid()
    return fig, ax

def plotMotorFault(motor):
    fig, axs = plt.subplots(3, sharex='col')

    linethick = 2.5

    axs[0].plot(motor["time"]*1000, # second to mS
            motor["currant"],
            lw=linethick)
    setupAx(axs[0])
    axs[0].set_title('Courant')
    axs[0].set(ylabel='A')

    axs[1].plot(motor["time"]*1000, # second to mS
            motor["voltage"],
            lw=linethick)
    setupAx(axs[1])
    axs[1].set_title('Tension')
    axs[1].set(ylabel='V')

    axs[2].plot(motor["time"]*1000, # second to mS
            motor["speed"],
            lw=linethick)
    setupAx(axs[2])
    axs[2].set_title('Vitesse')
    axs[2].set(ylabel='rad/s')

    for ax in axs.flat:
        ax.set(xlabel='Time (ms)')
        ax.set_xlim(0, 2000)
        ax.grid(True)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    return fig, axs

def plotMotorsCurrant(motors):
    fig, ax = plotMotors(motors, "currant")
    ax.set_xlabel('Temps (ms)')
    ax.set_ylabel('Courant (A)')
    plt.title('Courbes de courant des moteurs')
    plt.savefig(path.join(FIGURE_CHAP1_FOLDER, 'courant_moteurs.png'))
    plt.show()

def plotMotorsSpeed(motors):
    fig, ax = plotMotors(motors, "speed")
    ax.set_xlabel('Temps (ms)')
    ax.set_ylabel('Vitesse (rad/s)')
    plt.title('Courbes de vitesse des moteurs')
    plt.savefig(path.join(FIGURE_CHAP1_FOLDER, 'vitesse_moteurs.png'))
    plt.show()


def motorFault(motors, fault):
    i = 0
    for m in motors:
        fig, axs = plotMotorFault(m)
        #fig.suptitle('Simulation du moteur PMDC avec: '+ FAULT_TYPE[fault["fault"][i]], y=1)
        fig.set_size_inches(7.2 ,4.45*1.2)
        #fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.savefig(path.join(FIGURE_CHAP1_FOLDER, 'moteur-{}_faute-{}.png'.format(fault["motor"][i],fault["fault"][i])))
        plt.show()
        i+=1

if __name__ == "__main__":
    motorDataset = MotorFaultDataset(path.join(SIMULATION_MOTOR_FOLDER, "result.csv"), SIMULATION_MOTOR_FOLDER)

    Y = motorDataset.getMotorsData()

    motor1Data = Y[Y["motor"] == 1]
    motor1Idx = Y["id"].to_numpy()

    data = motorDataset[88]
    time = range(0, 5001)
    plt.plot(time, data["data"][SPEED_IDX,:])

    i=0
