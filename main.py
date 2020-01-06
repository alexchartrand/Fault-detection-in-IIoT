import torch
import torch.nn as nn

import argparse
import numpy as np
import torchvision.transforms
import matplotlib.pyplot as plt
from Data_manipulation.DataLoader import createDataLoader
import Data_manipulation.DataTransform as DataTransform
import  Data_manipulation.DataNormalization as DataNormalization
from Models.FCNNs import FCNNs
from os import path, listdir
from constant import *
import re

cuda = torch.cuda.is_available()
if cuda:
    print("Running model on GPU")
else:
    print("Running model on CPU")

## Sets hyper_param
data_size = (4,5001)
number_of_class = 6
number_of_sensors = 4
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

def trainModel(args, train_loader, valid_loader):
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0

    learning_curve_nll_train = list()
    learning_curve_nll_valid = list()
    learning_curve_acc_train = list()
    learning_curve_acc_valid = list()

    best_acc = -np.inf

    if args.model == "FCNN":
        model = FCNNs(number_of_sensors, number_of_class)
    else:
        raise Exception("Unknown model type")

    if cuda:
        model = model.cuda()

    optimizer = model.getOptimizer()

    for e in range(args.epoch):
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

        if args.save_best:
            if round(valid_acc, 3) > best_acc:
                best_acc = round(valid_acc, 3)
                torch.save(model.state_dict(), path.join(SAVED_MODEL_FOLDER, f'{args.model}_acc_{best_acc}.pth'))
                print('saved model')

        print(" [NLL] TRAIN {} / VALIDATION {}".format(
            train_loss, valid_loss))
        print(" [ACC] TRAIN {} / VALIDATION {}".format(
            train_acc, valid_acc))

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(learning_curve_nll_train, label='train')
    ax1.plot(learning_curve_nll_valid, label='validation')
    ax1.legend(bbox_to_anchor=(1, 1), loc=2)
    ax1.set_title('Negative Log_likelihood')
    ax1.set_xlabel('Epoch')

    ax2.plot(learning_curve_acc_train, label='train')
    ax2.plot(learning_curve_acc_valid, label='validation')
    ax2.legend(bbox_to_anchor=(1, 1), loc=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()

    return model

def loadModel(modelType):
    if modelType == "FCNN":
        model = FCNNs(number_of_sensors, number_of_class)
    else:
        raise Exception("Unknown model type")

    if cuda:
        model = model.cuda()

    modelPath = selectBestModel(modelType)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    return model

def selectBestModel(modelType):
    regex = r"{}_acc_(\d+\.\d+)\.pth".format(modelType)
    modelPath = ""
    maxAccuracy = 0.0
    modelList = [path.join(SAVED_MODEL_FOLDER, f) for f in listdir(SAVED_MODEL_FOLDER)
     if path.isfile(path.join(SAVED_MODEL_FOLDER, f)) and re.search(regex, f, re.IGNORECASE)]

    for model in modelList:
        matches = re.search(regex, model, re.IGNORECASE)
        acc = float(matches.group(1))
        if acc > maxAccuracy:
            modelPath = model

    return modelPath

def main(args):
    motor_transforms = torchvision.transforms.Compose([
        DataTransform.ToTensor(),
        DataNormalization.NoramalizeMinMax()])

    train_loader, valid_loader, test_loader = createDataLoader(args.data_path, args.batch_size, motor_transforms)

    if args.train:
        model = trainModel(args, train_loader, valid_loader)
    elif args.eval:
        model = loadModel(args.model)

        # Evaluate model
        test_loss = evaluate(model, test_loader, criterion)
        test_acc = evaluate(model, test_loader, accuracy)

        print("Model evaluation ===================")
        print("Test accuracy: ", str(test_acc))
        print("Test loss: ", str(test_loss))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IoT Simulation')

    parser.add_argument('--model', type=str, default='FCNN',
                        help='Model selection', choices=['FCNN'])
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epoch')
    parser.add_argument('--train', action='store_true',
                        help='Set model to training')
    parser.add_argument('--save_best', action='store_true',
                        help='Save the best model while training')
    parser.add_argument('--eval', action='store_true',
                        help='Set model to evaluation')
    parser.add_argument('--data_path', help='Path to data folder')

    args = parser.parse_args()

    main(args)