from constant import *
from Plotting.PlotSetup import *
import itertools
from os import path
import pickle

model_name = 'FCNN'
normalization = 'none'

def plot_curve(acc_train, acc_valid, nll_train, nll_valid):
    alphaVal = 0.75
    linethick = 2.5

    xticks = list(range(0, len(acc_train), 2))

    f, axs = plt.subplots(2, 1, sharex='col')
    axs[0].plot(nll_train, label='Train', color=colourWheel[0], alpha=alphaVal, lw=linethick)
    axs[0].plot(nll_valid, label='Validation', color=colourWheel[1], alpha=alphaVal, lw=linethick)
    axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right', ncol=1, handlelength=4)
    #axs[0].set_ylim(0, 15)
    axs[0].set_title('Negative Log likelihood')

    axs[1].plot(acc_train, label='Train', color=colourWheel[0], alpha=alphaVal, lw=linethick)
    axs[1].plot(acc_valid, label='Validation', color=colourWheel[1], alpha=alphaVal, lw=linethick)
    #axs[1].legend( frameon=False, loc='upper left', ncol=1, handlelength=4) #bbox_to_anchor=(1, 1),
    #axs[1].set_ylim(0, 1)
    axs[1].set_title('Accuracy')

    for ax in axs.flat:
        ax.set(xlabel='Epoch')
        #ax.set_xlim(0, 20)
        ax.grid(True)
        ax.set_xticks(xticks)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(path.join(FIGURE_CLASSIFICATION_FOLDER, f'{model_name}_{normalization}_train_vs_valid.png'))
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8, 6)) #figsize=(8, 6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes)
    #plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path.join(FIGURE_CLASSIFICATION_FOLDER, f'{model_name}_{normalization}_confusion.png'))
    plt.show()

def plotTrainVsValid():

    with open(path.join(ROOT_DIR, SAVED_CURVE_FOLDER, normalization, f'{model_name}_learning_curve_nll_train.pkl'), 'rb') as fp:
        learning_curve_nll_train = pickle.load(fp)

    with open(path.join(ROOT_DIR, SAVED_CURVE_FOLDER, normalization, f'{model_name}_learning_curve_nll_valid.pkl'), 'rb') as fp:
        learning_curve_nll_valid = pickle.load(fp)

    with open(path.join(ROOT_DIR, SAVED_CURVE_FOLDER, normalization, f'{model_name}_learning_curve_acc_train.pkl'), 'rb') as fp:
        learning_curve_acc_train = pickle.load(fp)

    with open(path.join(ROOT_DIR, SAVED_CURVE_FOLDER, normalization, f'{model_name}_learning_curve_acc_valid.pkl'), 'rb') as fp:
        learning_curve_acc_valid = pickle.load(fp)

    plot_curve(learning_curve_acc_train, learning_curve_acc_valid, learning_curve_nll_train, learning_curve_nll_valid)

if __name__ == "__main__":
    plotTrainVsValid()