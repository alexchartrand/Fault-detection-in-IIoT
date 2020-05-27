from Plotting.PlotSetup import *
import itertools

def plot_curve(acc_train, acc_valid, nll_train, nll_valid):
    alphaVal = 0.75
    linethick = 2.5

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(nll_train, label='train', color=colourWheel[0], alpha=alphaVal, lw=linethick)
    ax1.plot(nll_valid, label='validation', color=colourWheel[1], alpha=alphaVal, lw=linethick)
    ax1.legend(bbox_to_anchor=(1, 1), loc=2)
    ax1.set_title('Negative Log likelihood')
    ax1.set_xlabel('Epoch')

    ax2.plot(acc_train, label='train', color=colourWheel[0], alpha=alphaVal, lw=linethick)
    ax2.plot(acc_valid, label='validation', color=colourWheel[1], alpha=alphaVal, lw=linethick)
    ax2.legend(bbox_to_anchor=(1, 1), loc=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    else:
        1  # print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()