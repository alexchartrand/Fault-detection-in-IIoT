import torch
import torch.nn as nn

import argparse
import numpy as np
import torchvision.transforms
from Data_manipulation.DataLoader import createTrainDataLoader, createTestDataLoader
import Data_manipulation.DataTransform as DataTransform
import  Data_manipulation.DataNormalization as DataNormalization
from Models.FCNNs import FCNNs
from os import path, listdir
from constant import *
import re
import pickle
from Plotting.PlotResult import plot_curve

#%matplotlib inline

cuda = torch.cuda.is_available()
if cuda:
    print("Running model on GPU")
else:
    print("Running model on CPU")

## Sets hyper_param
data_size = (4,5001)
number_of_class = 6
number_of_sensors = 3
store_every = 200
criterion = nn.CrossEntropyLoss() # to compute the loss

def accuracy(proba, y):
    res = proba.max(1)[1]
    correct = torch.eq(res, y).sum().float()
    return correct / y.size(0)

def evaluate(model, dataset_loader):
    LOSSES = 0
    COUNTER = 0
    correct = 0
    model.eval()
    for batch in dataset_loader:

        x = batch['data']
        y = batch['fault']
        y = y[:, 1].long()  # Select the fault dimension

        if cuda:
            x = x.cuda()
            y = y.cuda()

        n = y.size(0)

        outputs = model(x)
        loss = criterion(outputs, y)
        res = outputs.max(1)[1]
        correct += torch.eq(res, y).sum().item()

        #LOSSES += loss.sum().data.cpu().numpy()
        LOSSES += loss.item()
        COUNTER += n

    model.train()
    return correct/float(COUNTER) * 100, LOSSES/float(COUNTER)

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
    model.train()
    optimizer = model.getOptimizer()

    print(f'Running model: {args.model} with {args.epoch} epochs')

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
            LOSSES += loss.item()
            COUNTER += n
            ITERATIONS += 1

            if ITERATIONS%(store_every/n) == 0:
                avg_loss = LOSSES / float(COUNTER)
                LOSSES = 0
                COUNTER = 0
                print(" Iteration {}: TRAIN {}".format(
                    ITERATIONS, avg_loss))

        train_acc, train_loss = evaluate(model, train_loader)
        valid_acc, valid_loss = evaluate(model, valid_loader)

        learning_curve_acc_train.append(train_acc)
        learning_curve_nll_train.append(train_loss)
        learning_curve_acc_valid.append(valid_acc)
        learning_curve_nll_valid.append(valid_loss)

        if args.save_best:
            if round(valid_acc, 3) > best_acc:
                best_acc = round(valid_acc, 3)
                torch.save(model.state_dict(), path.join(SAVED_MODEL_FOLDER, f'{args.model}_acc_{best_acc}_bsize_{args.batch_size}.pth'))
                print('saved model')

        print(" [NLL] TRAIN {} / VALIDATION {}".format(
            train_loss, valid_loss))
        print(" [ACC] TRAIN {} / VALIDATION {}".format(
            train_acc, valid_acc))

    with open(path.join(SAVED_CURVE_FOLDER, f'{args.model}_learning_curve_nll_train.pkl'), 'wb') as fp:
        pickle.dump(learning_curve_nll_train, fp)

    with open(path.join(SAVED_CURVE_FOLDER, f'{args.model}_learning_curve_nll_valid.pkl'), 'wb') as fp:
        pickle.dump(learning_curve_nll_valid, fp)

    with open(path.join(SAVED_CURVE_FOLDER, f'{args.model}_learning_curve_acc_train.pkl'), 'wb') as fp:
        pickle.dump(learning_curve_acc_train, fp)

    with open(path.join(SAVED_CURVE_FOLDER, f'{args.model}_learning_curve_acc_valid.pkl'), 'wb') as fp:
        pickle.dump(learning_curve_acc_valid, fp)

    return model

def loadModel(modelType, model_path):
    if modelType == "FCNN":
        model = FCNNs(number_of_sensors, number_of_class)
    else:
        raise Exception("Unknown model type")

    if cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(model_path))

    print(f'Model loaded: {model_path}')
    return model

def main(args):
    motor_transforms = torchvision.transforms.Compose([
        DataTransform.ToTensor(),
        DataNormalization.NoramalizeMinMax()])

    if args.train:
        train_loader, valid_loader = createTrainDataLoader(args.data_path, args.batch_size, motor_transforms, args.use_cache)
        print(f'Train size: {len(train_loader) * args.batch_size}, Valid size: {len(valid_loader) * args.batch_size}')
        model = trainModel(args, train_loader, valid_loader)
    elif args.eval:
        model = loadModel(args.model, args.model_path)
        #test_loader = createTestDataLoader(args.data_path, args.batch_size, motor_transforms)
        train_loader, valid_loader = createTrainDataLoader(args.data_path, args.batch_size, motor_transforms,
                                                           args.use_cache)
        print(f'Test size: {len(valid_loader) * args.batch_size}')
        # Evaluate model
        test_acc, test_loss = evaluate(model, valid_loader)

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
    parser.add_argument('--model_path', type=str, required=False, help='Path for the model to load')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use data caching in dataloader')

    args = parser.parse_args()

    main(args)