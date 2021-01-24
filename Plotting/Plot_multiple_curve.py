from constant import *
from Plotting.PlotSetup import *
import itertools
from os import path
import pickle

CURVES_PATH = [
    path.join(ROOT_DIR, SAVED_CURVE_FOLDER, 'Encoder_learning_curve_acc_valid.pkl'),
    path.join(ROOT_DIR, SAVED_CURVE_FOLDER, "saved_curve_dropout", 'Encoder_learning_curve_acc_valid.pkl'),
    path.join(ROOT_DIR, SAVED_CURVE_FOLDER, "saved_curve_dim", 'Encoder_learning_curve_acc_valid.pkl'),
    path.join(ROOT_DIR, SAVED_CURVE_FOLDER, "saved_curve_kernel", 'Encoder_learning_curve_acc_valid.pkl')
]

CURVES_NAME = [
    "Base",
    "Dropout",
    "Dimension",
    "Kernel"
]
OUTPUT_FNAME = 'Encoder_acc_param.png'

def plotTrainVsValid(curves, curves_name, out_fname):
    alphaVal = 0.75
    linethick = 2.5

    xticks = list(range(0, 30, 2))
    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch')
    ax.set(ylabel='Accuracy')
    ax.set_xlim(0, 29)
    ax.set_ylim(50, 90)
    ax.grid(True)
    ax.set_xticks(xticks)
    ax.set_title("Encoder hyper-parameters tuning")

    for i in range(len(curves)):
        with open(curves[i], 'rb') as fp:
            curve_data = pickle.load(fp)
        ax.plot(curve_data, label=curves_name[i], color=colourWheel[i], alpha=alphaVal, lw=linethick)

    ax.legend()

    plt.tight_layout()
    plt.savefig(path.join(FIGURE_CLASSIFICATION_FOLDER, out_fname))
    plt.show()

if __name__ == "__main__":
    plotTrainVsValid(CURVES_PATH, CURVES_NAME, OUTPUT_FNAME)