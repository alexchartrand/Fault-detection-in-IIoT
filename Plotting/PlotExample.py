from Plotting.PlotSetup import *
from constant import  *

def saveMotor(data):
    fig, axs = plt.subplots(3, sharex='col')

    axs[0].plot(data[0] * 1000,  # second to mS
                data[1])
    setupAx(axs[0])
    axs[0].set_title('Courant')
    axs[0].set(ylabel='A')

    axs[1].plot(data[0] * 1000,  # second to mS
                data[3])
    setupAx(axs[1])
    axs[1].set_title('Tension')
    axs[1].set(ylabel='V')

    axs[2].plot(data[0] * 1000,  # second to mS
                data[2])
    setupAx(axs[2])
    axs[2].set_title('Vitesse')
    axs[2].set(ylabel='rad/s')

    for ax in axs.flat:
        ax.set(xlabel='Temps (ms)')
        ax.set_xlim(0, 5000)
        ax.grid(True)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout()
    plt.savefig(path.join(FIGURE_MODELISATION_FOLDER, 'sim-no-fault.png'))
    plt.show()

def saveCommand(data):
    plt.figure()

    plt.plot(data[0] * 1000,  # second to mS
                data[1])
    plt.title('Commande')
    plt.ylabel('Vitesse (rad/s)')
    plt.xlabel('Temps (ms)')
    plt.grid(True)
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(path.join(FIGURE_MODELISATION_FOLDER, 'sim-no-fault_cmd.png'))
    plt.show()

motor_name = path.join(SIMULATION_MOTOR_FOLDER, "Solver", "sim_no-fault.csv")
data = np.genfromtxt(motor_name, delimiter=",", skip_header=1)
data = np.transpose(data)
saveMotor(data)

time_array = np.arange(0, data.shape[1] * T, T)
cmd_array = np.ndarray(data.shape[1])
cmd_array[0:1500] = 10
cmd_array[1500:] = 20
command = np.vstack((time_array, cmd_array))
saveCommand(command)
