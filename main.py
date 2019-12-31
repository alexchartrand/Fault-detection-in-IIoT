import torch
import torch.nn as nn

import numpy as np
import torchvision.transforms
import matplotlib.pyplot as plt
from DataLoader import createDataLoader
import DataTransform
import  DataNormalization
from FCNNs import FCNNs
from os import path
from constant import *

cuda = torch.cuda.is_available()

## Sets hyper_param
data_size = (4,5001)
number_of_class = 6
number_of_sensors = 4
batch_size = 16  # mini_batch size
num_epochs = 10  # number of training epochs
store_every = 200
criterion = nn.CrossEntropyLoss() # to compute the loss

def accuracy(proba, y):
    res = proba.max(1)[1]
    correct = torch.eq(res, y).sum().float()
    return correct / y.size(0)

def evaluate(model, dataset_loader, eval_fn):
    LOSSES = 0
    COUNTER = 0
    model.eval()
    for batch in dataset_loader:

        x = batch['data']
        y = batch['fault']
        y = y[:, 1].long()  # Select the fault dimension

        if cuda:
            x = x.cuda()
            y = y.cuda()

        loss = eval_fn(model(x), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        COUNTER += n

    model.train()

    return LOSSES / float(COUNTER)

def trainModel(model, train_loader, valid_loader):
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    optimizer = model.getOptimizer()

    learning_curve_nll_train = list()
    learning_curve_nll_valid = list()
    learning_curve_acc_train = list()
    learning_curve_acc_valid = list()

    best_acc = -np.inf
    for e in range(num_epochs):
        print(f'============= EPOCH {e} ========================')
        for batch in train_loader:
            optimizer.zero_grad()

            x = batch['data']
            y = batch['fault']
            y = y[:,1].long() # Select the fault dimension

            if cuda:
                x = x.cuda()
                y = y.cuda()

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n
            ITERATIONS += 1

            if ITERATIONS%(store_every/n) == 0:
                avg_loss = LOSSES / float(COUNTER)
                LOSSES = 0
                COUNTER = 0
                print(" Iteration {}: TRAIN {}".format(
                    ITERATIONS, avg_loss))

        train_loss = evaluate(model, train_loader, criterion)
        learning_curve_nll_train.append(train_loss)
        valid_loss = evaluate(model, valid_loader, criterion)
        learning_curve_nll_valid.append(valid_loss)

        train_acc = evaluate(model, train_loader, accuracy)
        learning_curve_acc_train.append(train_acc)
        valid_acc = evaluate(model, valid_loader, accuracy)
        learning_curve_acc_valid.append(valid_acc)

        if round(valid_acc, 3) > best_acc:
            best_acc = round(valid_acc, 3)
            torch.save(model.state_dict(), path.join(SAVED_MODEL_FOLDER, f'FCNN_acc_{best_acc}.pth'))
            print('saved model')

        print(" [NLL] TRAIN {} / VALIDATION {}".format(
            train_loss, valid_loss))
        print(" [ACC] TRAIN {} / VALIDATION {}".format(
            train_acc, valid_acc))

    return learning_curve_nll_train, \
           learning_curve_nll_valid, \
           learning_curve_acc_train, \
           learning_curve_acc_valid,

def main():
    motor_transforms = torchvision.transforms.Compose([
        DataTransform.ToTensor(),
        DataNormalization.NoramalizeMinMax()])
    train_loader, valid_loader, test_loader = createDataLoader(batch_size, motor_transforms)

    model = FCNNs(number_of_sensors, number_of_class)

    if cuda:
        model = model.cuda()

    nll_train, nll_valid, acc_train, acc_valid = trainModel(model, train_loader, valid_loader)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(nll_train, label='train')
    ax1.plot(nll_valid, label='validation')
    ax1.legend(bbox_to_anchor=(1, 1), loc=2)
    ax1.set_title('Negative Log_likelihood')
    ax1.set_xlabel('Epoch')

    ax2.plot(acc_train, label='train')
    ax2.plot(acc_valid, label='validation')
    ax2.legend(bbox_to_anchor=(1, 1), loc=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')

    plt.tight_layout()

    # Evaluate model
    test_loss = evaluate(model, test_loader, criterion)
    test_acc = evaluate(model, test_loader, accuracy)

    print("Model evaluation ===================")
    print("Test accuracy: ", str(test_acc))
    print("Test loss: ", str(test_loss))



if __name__ == '__main__':
    main()