from constant import  *
from Data_manipulation.DataLoader import MotorFaultDataset
from Plotting.PlotSetup import *

def showFault(fault):
    fault = fault.groupby("fault").size().values
    faultName = list(FAULT_TYPE_SHORT.values())
    totalFault = fault.sum()
    fig, ax = plt.subplots()

    bars = ax.bar(faultName, fault/totalFault*100)

    for rect in bars:
        height = rect.get_height()
        ax.annotate('{} %'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title("Occurence des fautes (%)")

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    motorDataset = MotorFaultDataset(path.join(SIMULATION_MOTOR_FOLDER, "result.csv"), SIMULATION_MOTOR_FOLDER)

    Y = motorDataset.getMotorsData()

    showFault(Y)

    motor1Data = Y[Y["motor"] == 1]
    motor1Idx = Y["id"].to_numpy()

    i=0