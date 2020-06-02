from os import path
import numpy as np
from constant import  *
from Data_manipulation.DataLoader import MotorFaultDataset
import Data_manipulation.DataNormalization as norm
import Data_manipulation.DataTransform as tf
from Plotting.PlotSetup import *
from torchvision import transforms


def plotMotors(motors, yValueName):
    fig, ax = plt.subplots()

    alphaVal = 0.75
    timeLimit = 1000

    for i in range(len(motors)):
        m = motors[i]
        m = m[:][0:timeLimit]
        ax.plot(m["time"]*1000, # second to mS
                m[yValueName],
                color=colourWheel[i % len(colourWheel)],
                linestyle='-',
                dashes=dashesStyles[i % len(dashesStyles)],
                label="Motor {}".format(i+1),
                alpha=alphaVal)

    setupAx(ax)
    ax.legend(frameon=False, loc='upper right', ncol=1, handlelength=4)
    plt.grid()
    return fig, ax

def plotMotor(motor):
    fig, axs = plt.subplots(3, sharex='col')

    axs[0].plot(motor[3]*1000, # second to mS
            motor[CURRANT_IDX])
    setupAx(axs[0])
    axs[0].set_title('Courant')
    axs[0].set(ylabel='A')

    axs[1].plot(motor[3]*1000, # second to mS
            motor[VOLTAGE_IDX])
    setupAx(axs[1])
    axs[1].set_title('Tension')
    axs[1].set(ylabel='V')

    axs[2].plot(motor[3]*1000, # second to mS
            motor[SPEED_IDX])
    setupAx(axs[2])
    axs[2].set_title('Vitesse')
    axs[2].set(ylabel='rad/s')

    for ax in axs.flat:
        ax.set(xlabel='Time (ms)')
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
        fig, axs = plotMotor(m)
        #fig.suptitle('Simulation du moteur PMDC avec: '+ FAULT_TYPE[fault["fault"][i]], y=1)
        fig.set_size_inches(7.2 ,4.45*1.2)
        #fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.savefig(path.join(FIGURE_CHAP1_FOLDER, 'moteur-{}_faute-{}.png'.format(fault["motor"][i],fault["fault"][i])))
        plt.show()
        i+=1

if __name__ == "__main__":
    transform = transforms.Compose([tf.ToTensor(), tf.Derivative(), tf.ToNumpy()])
    motorDatasetTransform = MotorFaultDataset(csv_file=path.join(SIMULATION_MOTOR_FOLDER, "simulation", "result.csv"),
                                              root_dir=path.join(SIMULATION_MOTOR_FOLDER,"simulation"),
                                              transform=transform)
    motorDataset = MotorFaultDataset(csv_file=path.join(SIMULATION_MOTOR_FOLDER, "simulation", "result.csv"),
                                              root_dir=path.join(SIMULATION_MOTOR_FOLDER,"simulation"))

    data = motorDatasetTransform.getPlotableData(2340)

    fig, axs = plotMotor(data)
    for ax in axs.flat:
        ax.set_xlim(0, 5000)
        ax.set_ylim(-1.5, 1.5)

    fig.tight_layout()
    plt.show()

    Y = motorDatasetTransform.getMotorsData()

